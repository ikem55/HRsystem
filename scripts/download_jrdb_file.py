from modules.jra_jrdb_download import JrdbDownload

jrdb = JrdbDownload()
jrdb.procedure_download()

jrdb.move_file()