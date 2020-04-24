import pickle
import pandas as pd


# lightGBMの特徴量をチェックする。不要な項目があれば削除する。

pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_rows', 300)

model_folder = "E:\python/for_test_model/lbr_v1/race_lgm/"
this_model_name = "race_lgm_主催者コード_2_UMAREN_ARE"

with open(model_folder + this_model_name + '.pickle', 'rb') as f:
    model = pickle.load(f)
    importance = pd.DataFrame({"特徴": model.feature_name(), "重要度": model.feature_importance()}).sort_values(["重要度", "特徴"], ascending=False)
    print(importance.head(300))
    print(importance.tail(300))
