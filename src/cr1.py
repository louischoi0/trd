from cr import get_connection
import sys
from pykrx import stock

if __name__ == "__main__":
    name = '코스피'
    ticker = '1001'
    sdate, edate = sys.argv[1:]
    con = get_connection()

    df = stock.get_index_ohlcv(sdate, edate, ticker, 'd')
    c = [ 'open', 'high', 'low', 'end', 'volume', 'tot', 'change']
    df.columns = c
    df.index.name = 'date'
    df['change'] = df['end'].pct_change() * 100


    df['ticker'] = ticker
    df['name'] = name
    df["tot"] /= 10000 * 10000
    df["volume"] /= 10000
    df.iloc[1:, :].to_sql('stpr', if_exists='append', con = con, index=True)
    print(df)
