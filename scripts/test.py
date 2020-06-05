from modules.jrdb_download import JrdbDownload

jrdb = JrdbDownload()
#jrdb.procedure_download()

#jrdb.move_file()
#paci_df = jrdb.get_jrdb_page("PACI")
#print(paci_df.shape)
#print(paci_df.tail())

"""
from modules.jra_sk_model import JRASkModel

MODEL_VERSION = 'jra_rc_raptype'
MODEL_NAME = 'race_lgm'
start_date = '2019/01/01'
end_date = '2019/12/31'
mock_flag = False
test_flag = False
mode = 'predict'
sk_model = JRASkModel(MODEL_NAME, MODEL_VERSION, start_date, end_date, mock_flag, test_flag, mode)

start_date = sk_model.get_recent_day(start_date)
print(start_date)
"""
