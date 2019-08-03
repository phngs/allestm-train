"""

"""
import argparse
import sys
import logging
import sqlite3
from multiprocessing import Pool
import itertools
import tempfile
import inspect
import random

import numpy as np
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, concatenate, Bidirectional, LSTM, BatchNormalization, Activation, add
from keras.models import load_model
from keras.optimizers import Adam

from mllib.db import db_store_predicted, db_create_experiments_table, db_model_exists, get_num_folds, \
    get_ids_and_lengths, get_max_length, db_store_model, db_get_loss
from mllib.features.utils import name_to_feature
from mllib.retrievers import SQLRetriever


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Dataset:
    """
    Dataset container.
    """
    def __init__(self):
        """
        Constructor initializing the properties.
        """
        self.ids = []
        self.lengths = []
        self.emb_values = []
        self.loc_values = []
        self.cont_values = []
        self.bin_values = []
        self.cat_values = []
        self.weights = []


def parse_args(argv):
    """
    Parse the arguments.

    Args:
        argv: List of command line args.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('db_file', help='Sqlite3 database file.')
    parser.add_argument('dataset_name', help='Name of the dataset in the DB.')
    parser.add_argument('experiment', help='Name of the experiment.')

    parser.add_argument('-p', '--params', default=[], action='append', help='Param list in the form of param:val1,val2,...')

    parser.add_argument('-e', '--emb_features', default=[], action='append', help='Names of embedding features')
    parser.add_argument('-l', '--loc_features', default=[], action='append', help='Names of local features')

    parser.add_argument('-c', '--cont_targets', default=[], action='append', help='Names of continuous targets')
    parser.add_argument('-b', '--bin_targets', default=[], action='append', help='Names of binary targets')
    parser.add_argument('-a', '--cat_targets', default=[], action='append', help='Names of categorical targets')

    parser.add_argument('-i', '--limit', type=int, default=None, help='Number of proteins to use (for testing purposes).')

    return parser.parse_args(argv)


def _calc_feature_values(args):
    """
    Helper function for parallelization of the feature calculation process.

    Args:
        args: List containing db_file, feature (class instance), and the identifier of the protein.

    Returns:
        NumPy Array with the shape (protein_length, feature_values).
    """
    db_file, feature, identifier = args
    conn = sqlite3.connect(db_file, timeout=120)
    retriever = SQLRetriever(conn, feature.query)
    result = feature.fit_transform(retriever.transform(identifier))
    conn.close()
    print('.', end='', flush=True)
    return result


def calc_feature_values(db_file, identifiers, max_length, feature):
    """
    Calculate the given feature for a list of identifiers.

    Args:
        db_file: The sqlite3 DB file.
        identifiers: List of identifiers.
        max_length: Maximum length of sequences (has impact of the resulting NumPy array).
        feature: Feature as class instance.

    Returns:
        NumPy array with the shape (num_of_identifiers, max_length, feature_values).
    """
    # Init the result array.
    feature_values = np.zeros((len(identifiers), max_length, len(feature)))

    # Ugly creating the input for the Pool.map function.
    task_args = zip(
        itertools.repeat(db_file, len(identifiers)),
        itertools.repeat(feature, len(identifiers)),
        identifiers
    )

    with Pool() as pool:
        pool_results = pool.map(_calc_feature_values, task_args)

    # Copy the result from the parallel processing to the result array
    # (where the shorter sequences are filled up with zeros).
    for i, pool_result in enumerate(pool_results):
        feature_values[i, :len(pool_result), :] = pool_result
    print()

    return feature_values


def create_dataset(db_file, max_length, emb_features, loc_features, cont_targets, bin_targets, cat_targets, dataset_name, kind, fold, limit=None):
    """
    Create a dataset instance and fill it with the feature values.
    Args:
        db_file: Sqlite3 DB file.
        max_length: Maximum length of the proteins in that fold.
        emb_features: Embedding feature instances.
        loc_features: Local feature instances.
        cont_targets: Continuous target instances.
        bin_targets: Binary target instances.
        cat_targets: Categorical target instances.
        dataset_name: Name of the dataset.
        kind: Kind of dataset (train, valid, test).
        fold: Number of the fold.

    Returns:
        Filled dataset instance.
    """
    conn = sqlite3.connect(db_file, timeout=120)
    dataset = Dataset()
    dataset.ids, dataset.lengths = get_ids_and_lengths(conn, dataset_name, kind, fold, limit=limit)
    conn.close()

    for feature in emb_features:
        log.info(f'Calculating embedding feature {type(feature)}.')
        dataset.emb_values.append(calc_feature_values(db_file, dataset.ids, max_length, feature))

    for feature in loc_features:
        log.info(f'Calculating local feature {type(feature)}.')
        dataset.loc_values.append(calc_feature_values(db_file, dataset.ids, max_length, feature))

    for target in cont_targets:
        log.info(f'Calculating continuous target {type(target)}.')
        dataset.cont_values.append(calc_feature_values(db_file, dataset.ids, max_length, target))

    for target in bin_targets:
        log.info(f'Calculating binary target {type(target)}.')
        dataset.bin_values.append(calc_feature_values(db_file, dataset.ids, max_length, target))

    for target in cat_targets:
        log.info(f'Calculating categorical target {type(target)}.')
        dataset.cat_values.append(calc_feature_values(db_file, dataset.ids, max_length, target))

    # Weights.
    weights = np.zeros((len(dataset.ids), max_length))

    for i, length in enumerate(dataset.lengths):
        weights[i, :length] = 1

    for _ in cont_targets + bin_targets + cat_targets:
        dataset.weights.append(weights)

    return dataset


def build_model(params, max_length, emb_features, loc_features, cont_targets, bin_targets, cat_targets):
    """
    Build and compile the model.

    Args:
        params: Parameters to use.
        dataset: Any dataset to infer shapes.
        max_length: Maximum length of the sequences.
        emb_features: Embedding feature instances.
        loc_features: Local features instances.
        cont_targets: Continuous target instances.
        bin_targets: Binary target instances.
        cat_targets: Categorical target instances.

    Returns:
        Tupel containing the model and a dict with the custom objects used
        in the features.
    """

    """
    - Suitable feature activations.
    """
    use_embedding = params['use_embedding']
    num_layers = params['num_layers']
    layer_type = params['layer_type']
    layer_size = params['layer_size']
    output_dense_size = params['output_dense_size']
    window = params['window']
    dilations = params['dilations']
    use_residuals = params['use_residuals'] # add, concatenate
    num_dense_layers = params['num_dense_layers']
    lr = params['lr']

    optimizer = Adam(lr=lr)

    # Embedding inputs.
    emb_inputs = []
    emb_outputs = []
    for emb_feature in emb_features:
        emb_inputs.append(Input(shape=(max_length, len(emb_feature)), name=f'{type(emb_feature).__module__}.{type(emb_feature).__qualname__}'))
        if use_embedding > 0:
            emb_outputs.append(Conv1D(use_embedding, 1)(emb_inputs[-1]))
        else:
            emb_outputs.append(emb_inputs[-1])

    # Local/other inputs.
    loc_inputs = []
    for loc_feature in loc_features:
        loc_inputs.append(Input(shape=(max_length, len(loc_feature)), name=f'{type(emb_feature).__module__}.{type(loc_feature).__qualname__}'))

    # Merge them.
    layer = concatenate(emb_outputs + loc_inputs)

    # Stacked RNNs.
    for layer_i in range(num_layers):
        if use_residuals in ['add', 'concatenate']:
            if use_residuals == 'add' and layer_i == 0:
                if layer_type == 'lstm':
                    residuals = Conv1D(layer_size * 2, 1, padding='same')(layer)
                else:
                    residuals = Conv1D(layer_size, 1, padding='same')(layer)
            else:
                residuals = layer

        if layer_type == 'lstm':
            layer = Bidirectional(LSTM(layer_size, return_sequences=True))(layer)
        elif layer_type == 'cnn':
            layer = Conv1D(layer_size, window, padding='same')(layer)
            layer = Activation('relu')(layer)
        elif layer_type == 'dcnn':
            for dil_i in range(dilations + 1):
                layer = Conv1D(layer_size, window, dilation_rate=2**dil_i, padding='same')(layer)
                layer = Activation('relu')(layer)

        if use_residuals == 'add':
            layer = add([layer, residuals])
        elif use_residuals == 'concatenate':
            layer = concatenate([layer, residuals])

    # Last dense layer before the outputs.
    output = layer
    for _ in range(num_dense_layers):
        output = Conv1D(output_dense_size, 1)(output)
        output = Activation('relu')(output)

    # Final outputs.
    outputs = []
    for target in cont_targets + bin_targets + cat_targets:
        outputs.append(Conv1D(len(target), 1, activation=target.keras_activation)(output))

    model = Model(inputs=[*emb_inputs, *loc_inputs], outputs=outputs)

    # Gather losses from the features.
    losses = []
    for target in cont_targets + bin_targets + cat_targets:
        losses.append(target.keras_loss)

    # Compile.
    model.compile(optimizer=optimizer, loss=losses, sample_weight_mode='temporal')
    model.summary()

    return model


def train_model(params, model, train_data, valid_data, limit=None):
    """
    Train the model.

    Args:
        params: Parameters to use.
        model: The compiled model.
        train_data: Training dataset.
        valid_data: Validation dataset.
        limit: If true, do only one iteration.

    Returns:
        Trained model.
    """
    batch_size = params['batch_size']
    epochs = 200
    if limit:
        epochs = 1
    patience = 10
    patience_lr = 5

    with tempfile.NamedTemporaryFile() as model_file:
        callbacks = [
            EarlyStopping(patience=patience, verbose=1),
            ModelCheckpoint(model_file.name, save_best_only=True, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=patience_lr, min_lr=0.0001, verbose=1)
        ]

        history = model.fit(
            [*train_data.emb_values, *train_data.loc_values],
            [*train_data.cont_values, *train_data.bin_values, *train_data.cat_values],
            validation_data=([*valid_data.emb_values, *valid_data.loc_values],
                             [*valid_data.cont_values, *valid_data.bin_values, *valid_data.cat_values],
                             valid_data.weights),
            sample_weight=train_data.weights,
            epochs=epochs, batch_size=batch_size,
            shuffle=True, callbacks=callbacks)

        # Reset to state of best model.
        try:
            model = load_model(model_file.name)
        except OSError:
            return None, None

        return model, history.history


def eval_model(params, model, valid_data):
    """
    Evaluate the model.

    Args:
        params: Parameters to use.
        model: The compiled model.
        valid_data: Validation dataset.

    Returns:
        Loss.
    """
    batch_size = params['batch_size']

    loss = model.evaluate(
        [*valid_data.emb_values, *valid_data.loc_values],
        [*valid_data.cont_values, *valid_data.bin_values, *valid_data.cat_values],
        sample_weight=valid_data.weights,
        batch_size=batch_size)

    log.info(f'Losses on validation set: {loss}')
    if np.isscalar(loss):
        return loss
    else:
        return loss[0]


def test_model(model, test_data):
    """
    Run predictions.

    Args:
        model: Trained model.
        test_data: Test dataset.

    Returns:
        List of NumPy arrays containing predictions.
    """
    return model.predict([*test_data.emb_values, *test_data.loc_values])


def process_predictions(lengths, target, raw_predicted):
    """
    Removes padded positions from the predicted/observed values and
    calls inverse_transform if available.

    Args:
        lengths: Lengths of the entries in the dataset.
        target: Target feature instance.
        raw_predicted: Raw predicted/observed values.

    Returns:
        List of NumPy arrays: [(length, target_vals), ...].
    """
    predicted = []

    for i, raw_pred in enumerate(raw_predicted):
        pred = raw_pred[:lengths[i], :]
        if hasattr(target, 'inverse_transform'):
            pred = target.inverse_transform(pred)
        predicted.append(pred)

    return predicted


def main(argv):
    """
    Main method.

    Args:
        argv: Command line arguments.

    Returns:
        Nothing.
    """
    args = parse_args(argv)


    conn = sqlite3.connect(args.db_file, timeout=120)

    # Create the feature instances.
    emb_features = [name_to_feature(x) for x in args.emb_features]
    loc_features = [name_to_feature(x) for x in args.loc_features]
    cont_targets = [name_to_feature(x) for x in args.cont_targets]
    bin_targets = [name_to_feature(x) for x in args.bin_targets]
    cat_targets = [name_to_feature(x) for x in args.cat_targets]

    num_folds = get_num_folds(conn, args.dataset_name)

    log.info('Creating experiments table if not exists.')
    db_create_experiments_table(conn)

    params_map = {
        'use_embedding': [0],
        'num_layers': [1],
        'layer_type': ['lstm'],
        'layer_size': [128],
        'output_dense_size': [256],
        'window': [21],
        'batch_size': [4],
        'dilations': [1],
        'use_residuals': [0],
        'num_dense_layers': [1],
        'lr': [0.001]
    }

    for p in args.params:
        param, v = p.split(':')
        val_strs = v.split(',')
        vals = []
        for val_str in val_strs:
            if val_str == 'None':
                vals.append[None]
            elif val_str.isdigit():
                vals.append(int(val_str))
            else:
                try:
                    vals.append(float(val_str))
                except ValueError:
                    vals.append(val_str)
        params_map[param] = vals

    for iter in range(100):
        log.info(f'Iteration {iter + 1}')

        window = random.choice(params_map['window'])
        use_embedding = random.choice(params_map['use_embedding'])
        num_layers = random.choice(params_map['num_layers'])
        layer_type = random.choice(params_map['layer_type'])
        layer_size = random.choice(params_map['layer_size'])
        output_dense_size = random.choice(params_map['output_dense_size'])
        batch_size = random.choice(params_map['batch_size'])
        dilations = random.choice(params_map['dilations'])
        use_residuals = random.choice(params_map['use_residuals'])
        num_dense_layers = random.choice(params_map['num_dense_layers'])
        lr = random.choice(params_map['lr'])

        params = {'architecture': inspect.getsource(build_model),
                  'training': inspect.getsource(train_model),
                  'use_embedding': use_embedding,
                  'num_layers': num_layers,
                  'layer_type': layer_type,
                  'layer_size': layer_size,
                  'output_dense_size': output_dense_size,
                  'batch_size': batch_size,
                  'window': window,
                  'dilations': dilations,
                  'use_residuals': use_residuals,
                  'num_dense_layers': num_dense_layers,
                  'lr': lr
                  }

        improvements = []

        for fold in range(num_folds):
            log.info(f'Starting with fold {fold}.')

            prev_loss = None
            if db_model_exists(conn, args.experiment, fold):
                prev_loss = db_get_loss(conn, args.experiment, fold)
                log.info(f'Experiment {args.experiment} fold {fold} already exists with loss {prev_loss}.')

            max_length = get_max_length(conn, args.dataset_name, fold)

            log.info('Creating train dataset.')
            train_data = create_dataset(args.db_file, max_length, emb_features, loc_features, cont_targets, bin_targets, cat_targets, args.dataset_name, 'train', fold, limit=args.limit)

            log.info('Creating valid dataset.')
            valid_data = create_dataset(args.db_file, max_length, emb_features, loc_features, cont_targets, bin_targets, cat_targets, args.dataset_name, 'valid', fold, limit=args.limit)

            log.info('Building model.')
            model = build_model(params, max_length, emb_features, loc_features, cont_targets, bin_targets, cat_targets)

            log.info('Training model.')
            model, history = train_model(params, model, train_data, valid_data, limit=args.limit)

            if model is None:
                log.info(params)
                log.info('Model did not train properly, skipping.')
                continue

            valid_loss = eval_model(params, model, valid_data)

            del train_data

            improvements.append([fold, prev_loss, valid_loss, prev_loss is None or valid_loss < prev_loss])

            if prev_loss is None or valid_loss < prev_loss:
                log.info(f'Validation loss improved. Previous loss was {prev_loss}, new loss is {valid_loss}.')

                log.info('Predicting on valid data.')
                valid_raw_predicted = test_model(model, valid_data)

                log.info('Storing valid predictions.')
                valid_raw_observed = valid_data.cont_values + valid_data.bin_values + valid_data.cat_values
                for i, target in enumerate(cont_targets + bin_targets + cat_targets):
                    observed = process_predictions(valid_data.lengths, target, valid_raw_observed[i])
                    if len(cont_targets + bin_targets + cat_targets) == 1:
                        predicted = process_predictions(valid_data.lengths, target, valid_raw_predicted)
                    else:
                        predicted = process_predictions(valid_data.lengths, target, valid_raw_predicted[i])

                    db_store_predicted(conn, f'{args.experiment}_VALID_{fold}', target, valid_data.ids, predicted)
                    db_store_predicted(conn, 'observed', target, valid_data.ids, observed)

                del valid_data

                log.info('Creating test dataset.')
                test_data = create_dataset(args.db_file, max_length, emb_features, loc_features, cont_targets, bin_targets, cat_targets, args.dataset_name, 'test', fold, limit=args.limit)

                log.info('Predicting on test data.')
                raw_predicted = test_model(model, test_data)

                log.info('Storing test predictions.')
                raw_observed = test_data.cont_values + test_data.bin_values + test_data.cat_values
                for i, target in enumerate(cont_targets + bin_targets + cat_targets):
                    observed = process_predictions(test_data.lengths, target, raw_observed[i])
                    if len(cont_targets + bin_targets + cat_targets) == 1:
                        predicted = process_predictions(test_data.lengths, target, raw_predicted)
                    else:
                        predicted = process_predictions(test_data.lengths, target, raw_predicted[i])

                    db_store_predicted(conn, args.experiment, target, test_data.ids, predicted)
                    db_store_predicted(conn, 'observed', target, test_data.ids, observed)

                del test_data

                log.info('Creating independent_test dataset.')
                independent_test_data = create_dataset(args.db_file, max_length, emb_features, loc_features, cont_targets, bin_targets, cat_targets, args.dataset_name, 'independent_test', 0, limit=args.limit)

                log.info('Predicting on independent_test data.')
                independent_raw_predicted = test_model(model, independent_test_data)

                log.info('Storing independent_test predictions.')
                independent_raw_observed = independent_test_data.cont_values + independent_test_data.bin_values + independent_test_data.cat_values
                for i, target in enumerate(cont_targets + bin_targets + cat_targets):
                    observed = process_predictions(independent_test_data.lengths, target, independent_raw_observed[i])
                    if len(cont_targets + bin_targets + cat_targets) == 1:
                        predicted = process_predictions(independent_test_data.lengths, target, independent_raw_predicted)
                    else:
                        predicted = process_predictions(independent_test_data.lengths, target, independent_raw_predicted[i])

                    db_store_predicted(conn, f'{args.experiment}_IND_TEST_{fold}', target, independent_test_data.ids, predicted)
                    db_store_predicted(conn, 'observed', target, independent_test_data.ids, observed)

                del independent_test_data

                log.info('Storing model.')
                db_store_model(conn, args.db_file, (lambda m, f: m.save(f)), args.experiment, fold, model, history, params, valid_loss)
            else:
                log.info(f'Validation loss did not improve. Previous loss was {prev_loss}, new loss is {valid_loss}.')

        del params['architecture']
        del params['training']
        log.info(params)
        log.info('Improvements:')
        if len(improvements) > 0:
            avg_prev = sum([x[1] for x in improvements if x[1] is not None]) / len(improvements)
            avg_now = sum([x[2] for x in improvements]) / len(improvements)
            avg_delta = avg_prev - avg_now
            log.info(f'Average improvement:prev:{avg_prev:.3f}:now:{avg_now:.3f}:delta:{avg_delta:.3f}')
            for imp in improvements:
                log.info(imp)
        else:
            log.info('Length of improvements was zero.')

    conn.close()



if __name__ == '__main__':
    main(sys.argv[1:])
