import yfinance as yf
import numpy as np
from cr import get_connection

if __name__ == "__main__":
    con = get_connection()
    start_date = '2019-01-01'
    end_date = '2023-09-15'

    data = yf.download(['USDKRW=X'],start=start_date, end=end_date)
    data["diff"] = data["High"] - data["Low"]
    data["change"] = data["Adj Close"].pct_change()

    data["change"] *= 100

    data = data.fillna(0)

    data = data.drop('Volume',axis=1)
    data = data.drop('Adj Close',axis=1)
    print(data)

    data.columns = [ "open", "high", "low", 'close', "diff", "change" ] 
    data.index.name = 'date'

    print(data)
    data.to_sql('krcc', if_exists='append', con=con, index=True)
