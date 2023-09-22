from cr import get_connection, get_ticker_dict
from matplotlib import pyplot
import pandas as pd
import sys


if __name__ == "__main__":
    ticker, start, end = sys.argv[1:]
    
    con = get_connection() 
    q = f"""select * from stpr where
    ticker = '{ticker}'
    and date >= '{start}' and date < '{end}'
    order by date"""
    df = pd.read_sql(q, con=con)
    
    i = df['open'].values[0]
    iv = df['volume'].values[0]

    pyplot.plot(df['volume'] * 1000 / iv - 1000, label='v')
    pyplot.plot(df['change'] * 1000 , label='p')
    pyplot.legend()
    pyplot.show()
    
    print(df)

