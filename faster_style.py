import math
import numpy as np
import pandas  as pd
from original_style import STYLE_CONTROL_ELEMENTS_V1

def construct_style_matrices(
    df,
    BASE=10,
    apply_ratio=[1, 1, 1, 1],
    style_elements=STYLE_CONTROL_ELEMENTS_V1,
    add_one=True,
):
    models = np.unique(df[['model_a', 'model_b']].values)
    model_to_idx = {model:idx for idx,model in enumerate(models)}

    # set two model cols by mapping the model names to their int ids
    matchups = df[['model_a', 'model_b']].map(lambda x: model_to_idx[x]).values
    
    # model_a win -> 1.0, tie -> 0.5, model_b win -> 0.0
    labels = np.select(
        condlist=[df['winner'] == 'model_a', df['winner'] == 'model_b'],
        choicelist=[1.0, 0.0],
        default=0.5
    )


    n = matchups.shape[0]
    k = int(len(style_elements) / 2)



    X2 = np.zeros(shape=(n, 2*k))
    
    # all_features = pd.json_normalize(df["conv_metadata"]).values
    # print(all_features.shape)

    # style_vector = np.array(
    #     [
    #         df.conv_metadata.map(
    #             lambda x: x[element]
    #             if type(x[element]) is int
    #             else sum(x[element].values())
    #         ).tolist()
    #         for element in style_elements
    #     ]
    # )
    # print(style_vector.shape)
