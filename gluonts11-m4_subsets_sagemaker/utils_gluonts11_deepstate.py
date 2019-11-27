# DeepState
import numpy as np
import pandas as pd
import pprint

# mxnet
import mxnet as mx

# utils
from utils_gluonts10 import get_dataset

# gluonts
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator

# DeepStateEstunator
from gluonts.model.deepstate import DeepStateEstimator

#########################
### deepstate wrapper ###
def deep_state(seed=42, data="m4_quarterly", epochs=100, batches=50):
    
    mx.random.seed(seed)
    np.random.seed(seed)
    
    dataset = get_dataset(data)

    trainer = Trainer(
        ctx=mx.cpu(0),
#         ctx=mx.gpu(0),
        epochs=epochs,
        num_batches_per_epoch=batches,
        learning_rate=1e-3,
    )

    cardinality = int(dataset.metadata.feat_static_cat[0].cardinality)
    estimator = DeepStateEstimator(
        trainer=trainer,
        cardinality=[cardinality],
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
        use_feat_static_cat=True,

    )

    predictor = estimator.train(dataset.train)
    
#     predictor = estimator.train(training_data=dataset.train, 
#                                 validation_data=dataset.test)
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset.test, predictor=predictor, num_samples=100
    )

    agg_metrics, item_metrics = Evaluator()(
        ts_it, forecast_it, num_series=len(dataset.test)
    )
    metrics = ["MASE", "sMAPE", "MSIS", "wQuantileLoss[0.5]", "wQuantileLoss[0.9]"]
    output = {key: round(value, 8) for key, value in agg_metrics.items() if key in metrics}
    output["epochs"] = epochs
    output["seed"] = seed
    
    df=pd.DataFrame([output])
    pprint(output)
    
    return(df)
