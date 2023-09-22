from cr import get_connection, get_ticker_dict
import pandas as pd
from sys import exit
import numpy as np

def md_3(ticker,name, series, volume, thres=10, p=5, i=5,f=5):
    result = [] 

    for idx in range(series.index.size):
        if idx < 20: continue

        vs = series.iloc[idx:idx+p+i+f]
        v = vs.values
        if len(v) < p + i + f: break

        psr, isr, fsr = v[:p], v[p:p+i], v[p+i:p+i+f]
        psr, isr, fsr = np.prod(psr), np.prod(isr), np.prod(fsr) 

        if np.isnan(v).any():
            continue

        if (psr - 1) * 100 > thres and (psr -1) * 100 < 70:
            ps = series.iloc[idx-20:idx].values

            profit20 = np.prod(ps)
            if np.isnan(profit20): continue

            volm = volume.iloc[idx:idx+p+i+f].values
        
            if ( volm[0] + volm[1] + volm[2] ) * 3.0 > volm[3] + volm[4]: continue

            if profit20 < 1.1: continue
            if profit20 > 1.7: continue

            volm = np.round(volm, 2)

            result.append((ticker, name, vs.index[0], vs.index[-1], psr, isr, fsr, profit20, "volume", *volm))
            result.append((ticker, name, vs.index[0], vs.index[-1], psr, isr, fsr, profit20, "change", *v))
        
    return result


def md_2(ticker,name, series, volume, thres=20, p=10, i=10 ,f=10):
    result = [] 

    for idx in range(series.index.size):
        if idx < 20: continue

        vs = series.iloc[idx:idx+p+i+f]
        v = vs.values
        if len(v) < p + i + f: break

        psr, isr, fsr = v[:p], v[p:p+i], v[p+i:p+i+f]
        psr, isr, fsr = np.prod(psr), np.prod(isr), np.prod(fsr) 

        if np.isnan(v).any() or np.isnan(psr):
            continue

        if (isr - 1) * 100 > thres:
            ps = series.iloc[idx-20:idx].values
            profit20 = np.prod(ps)

            volm = volume.iloc[idx:idx+p+i+f].values

            volm = np.round(volm, 2)

            result.append((ticker, name, vs.index[0], vs.index[-1], psr, isr, fsr, profit20, "volume", *volm))
            result.append((ticker, name, vs.index[0], vs.index[-1], psr, isr, fsr, profit20, "change", *v))
        
    return result

def get_max_relay(series, condition):
    result = [] 

    first = None
    before = False
    cnt = 0
    acc = 1

    for date, change in series.items():

        if condition(change):
            if first is None:
                first = date 
            before = True
            cnt += 1
            acc *= (change/100) + 1
        else:
            if before and cnt > 1:
                result.append( (first, date, cnt, (acc-1)*100 ) )
            acc = 1
            first = None
            cnt = 0
            before = False

    return result


def get_volume_table(con):
    df = pd.read_sql("select * from stpr", con=con)
    df = df[ ~df['name'].isnull() ]
    df = df.pivot(index='date', columns='ticker', values='volume')
    return df

def get_price_table(con):
    df = pd.read_sql("select * from stpr", con=con)
    df = df[ ~df['name'].isnull() ]
    df = df.pivot(index='date', columns='ticker', values='change')
    df /= 100
    df += 1
    return df

def get_vp(con, capmin=1000, capmax=2000):
    last = '2023-09-10'
    start = '2020-01-02'

    q = f"""
    select stpr.date, stpr.ticker, stpr.name, stpr.open, stpr.high, stpr.low, stpr.end as close, stpr.change,
    cast(stpr.tot * 100 as float) / (mkcap.marketcap * 10000000) as vrate, mkcap.marketcap,
    krcc.diff cdiff, krcc.change cchange , ksp.change kchange
    from  stpr
    join mkcap on stpr.ticker = mkcap.ticker and stpr.date = mkcap.date
    join (select date, change from stpr where ticker ='1001' and date >= '{start}' and date < '{last}') ksp on ksp.date = stpr.date
    join krcc krcc on krcc.date = stpr.date
    where ( (mkcap.marketcap > {capmin} and mkcap.marketcap < {capmax}) or stpr.ticker = '005930') and stpr.date >= '{start}' and stpr.date <= '{last}'
    and mkcap.prefered = 'N'
    order by stpr.date
    """

    print(q)

    df = pd.read_sql(q, con=con)
    _df = df[ df["ticker"] == '005930' ] #ss
    print(_df)

    ksp = _df['kchange'].values
    crc = _df['cchange'].values

    df['pdiff'] = (df['high'] - df['low']) / df['open']

    df['change'] /= 100
    df['change'] += 1

    chg_df = df.pivot(index='date', columns='ticker', values='change')
    mkc_df = df.pivot(index='date', columns='ticker', values='marketcap')
    pdf_df = df.pivot(index='date', columns='ticker', values='pdiff')
    vrt_df = df.pivot(index='date', columns='ticker', values='vrate')

    data = []
    data.append( chg_df.values.T )
    data.append( mkc_df.values.T )
    data.append( pdf_df.values.T )
    data.append( vrt_df.values.T )
    
    d = np.array(data)
    print(crc)
    print(ksp)

    np.save('data/d__mc5.10.npy', d)
    np.save('data/cch_mc5.10.npy', crc)
    np.save('data/ksp_mc5.10.npy', ksp)
    print('save!')

    return chg_df, mkc_df, pdf_df, vrt_df


if False and __name__ == "__main__":
    con = get_connection()
    d = get_ticker_dict("20230913")
    thres = 2.4
    chg_df, mk, pdf, vrt = get_vp(con)
    exit(0)

    for ticker in chg_df.columns:
        if not ticker in d: continue

        result = get_max_relay(chg_df[ticker], lambda x: x > thres )
        result = ( ( ticker, d[ticker], *x, thres ) for x in result )
        res = pd.DataFrame(result, columns=["ticker", "name", "sdate", "edate", "cnt", "profit", "thres"])
        res.index.size and print(res)
        #res.to_sql('kpfh', if_exists='append', con = con, index=False)

if False and __name__ == "__main__":
    con = get_connection()
    d = get_ticker_dict("20230913")
    thres = 2.4

    df = pd.read_sql("select * from stpr", con=con)
    df = df[ ~df['name'].isnull() ]
    df = df.pivot(index='date', columns='ticker', values='change')
    df = df.rename(columns={'1001': 'kospi'}, inplace=False)

    for ticker in df.columns:
        if not ticker in d: continue

        result = get_max_relay(df[ticker], lambda x: x > thres )
        result = ( ( ticker, d[ticker], *x, thres ) for x in result )
        res = pd.DataFrame(result, columns=["ticker", "name", "sdate", "edate", "cnt", "profit", "thres"])
        res.index.size and print(res)
        #res.to_sql('kpfh', if_exists='append', con = con, index=False)

    print(df)

if __name__ == "__main__":
    con = get_connection()
    d = get_ticker_dict("20230913")
    pdf, vdf= get_vp(con, 1000, 2000)

    data = []
    for ticker in pdf.columns:
        if not ticker in d: continue
        name = d[ticker]
        res = md_3(ticker, name, pdf[ticker], vdf[ticker], 25, 5,5,5)

        data.extend(res)
        break

    df = pd.DataFrame(data)

    df.columns = ("ticker", "name", "sdate", "edate", "pchange", "ichange", "fchange","p20", "ctype", *( "c_" + str(x) for x in range(1,16)))
    #df["ichange"] = df["ichange"].apply( lambda x: max(0.9, x) )
    
    profit = np.prod(df["ichange"].values)
    profit2 = np.prod(df["fchange"].values)
    print(profit, profit2, profit*profit2)
    #df.to_sql('event_kps_mm555_3', if_exists='append', con = con, index=False)

