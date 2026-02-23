cd /home/tg22/remote-pycharm/Vis_TaCarla

export MAIN_PATH="/media/hdd/text_data/deneme_data"
export TOWN_FOLDER="Town12"
export OUT_DIR="pyspark_vis_out"
export PYSPARK_ENABLE=true

streamlit run _UI.py --server.headless true --server.address 127.0.0.1 --server.port 8505