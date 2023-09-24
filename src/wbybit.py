from pybit.unified_trading import HTTP, WebSocket
from time import sleep
import json
import pandas as pd
from sqlalchemy import create_engine
import json

def get_connection():
    DB_HOST = "wwhale-on-gpt.cg6x7yqwsa6m.ap-northeast-2.rds.amazonaws.com"
    DB_USER = "postgres"
    DB_PASS = "a!02040608"
    DB_NAME = "systr"

    q= f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
    engine = create_engine(q)
    return engine

if __name__ == "__main__":
    con = get_connection()

    with open(".sec.json", "r") as f :
        dd = json.load(f)
        ak = dd["bak"]
        sk = dd["bsk"]

    ws = WebSocket(
        channel_type="linear",
        testnet=False,
        api_key=ak,
        api_secret=sk,
    )
    
    def callback(msg):
        print(msg)
        _d = msg['data']
        del msg['data']
        d = { **msg, **_d}
        df = pd.DataFrame([d]) 
        df.to_sql('bybit_btcusdt_tick', con=con, if_exists='append', index=False)
        print(df)


    s = ws.ticker_stream(
            symbol="BTCUSDT",
            callback=callback, )
    print('hji')

    while True:
        sleep(1)
