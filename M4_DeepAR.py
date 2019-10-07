# python script of M4_weekly and DeepAR

# standard imports
import numpy as np
import pandas as pd

# json
import json

# gluon data
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas

# gluon imports
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator


# model imports
# from gluonts.model.deepstate import DeepStateEstimator
# from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.deepar import DeepAREstimator

if __name__ == "__main__":

    import mxnet as mx
    from pprint import pprint

    dataset = get_dataset("m4_hourly", regenerate=False)

    trainer = Trainer(
        ctx=mx.cpu(0),
        epochs=100,      # default: 100
        num_batches_per_epoch=50,      #default: 50
        learning_rate=1e-3,
        # hybridize=False,
    )

    # cardinality = int(dataset.metadata.feat_static_cat[0].cardinality)
    estimator = DeepAREstimator(
        trainer=trainer,
        # context_length=168,
        # cardinality=[cardinality],
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
    )

    predictor = estimator.train(dataset.train)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset.test, predictor=predictor, num_eval_samples=100
    )

    agg_metrics, item_metrics = Evaluator()(
        ts_it, forecast_it, num_series=len(dataset.test)
    )

    pprint(agg_metrics)
