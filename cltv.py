##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############################################################
# GÖREVLER
###############################################################

#######################################################################################################
#######################################################################################################
# GÖREV 1: Veriyi Hazırlama


import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # veya 'QtAgg'
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from xarray.util.generate_ops import inplace

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


           # 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("datasets/flo_data_20k.csv")
df = df_.copy()
df.head()
df.shape
df.isnull().sum()
df.info()
    #######################################################################################################
    #######################################################################################################
           # 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)

    if (dataframe[variable] > up_limit).any():
        print("***************  OUTLİER !!  ****************")
        print(f"Üst eşik değerini aşan değişken: {variable} \nBelirlenen üst eşik değeri: {up_limit}")

    if (dataframe[variable] < low_limit).any():
        print("***************  OUTLİER !!  ****************")
        print(f"Alt eşik değerinin altında kalan değişken: {variable} \nBelirlenen alt eşik değeri: {low_limit}")

    dataframe.loc[dataframe[variable] > up_limit, variable] = round(up_limit)
    #dataframe.loc[dataframe[variable] < low_limit, variable] = round(low_limit)

           # Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
    #######################################################################################################
    #######################################################################################################
           # 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
           # aykırı değerleri varsa baskılayanız.
    df.describe().T
    df.info()
    # float64


    for col in df.columns:
        if df[col].dtypes in ["float64"]:
            plt.boxplot(df[col])
            plt.title(col)
            plt.show(block=True)

    for col in df.columns:
        if df[col].dtypes in ["float64"]:
            replace_with_thresholds(df, col)
            plt.boxplot(df[col])
            plt.title(col)
            plt.show(block=True)
    # lowerband altında bir outlier yok bundan dolayı alt limitde outlier yapmak istemiyorum
    #######################################################################################################
    #######################################################################################################
           # 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
    df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df.head()
    df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
    df.head()
           # 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.info()

for col in df.columns:
    if "date" in col.lower():
        df[col] = df[col].apply(pd.to_datetime)# böylede olur df[col].astype("datetime64[ns]")

df.info()
    #######################################################################################################
    #######################################################################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
           # 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
df["last_order_date"].max()
today = df["last_order_date"].max() + pd.Timedelta(days=2)

"""
Böylede yapılabilir
today_date = dt.datetime(2010,12,11)
type(today_date)
"""
    #######################################################################################################
    #######################################################################################################
           # 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
           # Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.
# Lifetime Veri Yapısının Hazırlanması
# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç
df.head()
df.info()


df["recency"] = round(((df["last_order_date"] - df["first_order_date"]).dt.days) / 7)
df["T"] = round(((today - df["first_order_date"]).dt.days)/7)


cltv = df[["recency","T","order_num_total","customer_value_total"]]
cltv.columns = ['recency', 'T', 'frequency', 'monetary']

cltv = cltv[cltv["frequency"] > 1]
cltv["monetary"] = cltv["monetary"] /  cltv["frequency"]


# sağlaması
"""
cltv = df.groupby("master_id").agg({
    "first_order_date":lambda x: x.min(),
    "last_order_date": lambda x: x.max(),
    "order_num_total": lambda x: x.sum(),
    "customer_value_total": lambda x: x.sum()
})

cltv["recency"] = round(((cltv["last_order_date"] - cltv["first_order_date"]).dt.days) / 7)
cltv["T"] = round((((today - cltv["first_order_date"]).dt.days)/7))


cltv = cltv[["recency","T","order_num_total","customer_value_total"]]

cltv.columns = ['recency', 'T', 'frequency', 'monetary']

cltv = cltv[cltv["frequency"] > 1]

cltv["monetary"] = cltv["monetary"] /  cltv["frequency"]

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv['frequency'],
        cltv['recency'],
        cltv['T'])


cltv[cltv['recency'] > cltv['T']]
"""


# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
           # 1. BG/NBD modelini fit ediniz.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv['frequency'],
        cltv['recency'],
        cltv['T'])
                # a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
cltv["exp_sales_3_month"] = bgf.predict(4*3,
                                              cltv['frequency'],
                                              cltv['recency'],
                                              cltv['T'])
                # b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv["exp_sales_6_month"] = bgf.predict(4*6,
                                              cltv['frequency'],
                                              cltv['recency'],
                                              cltv['T'])
           # 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv['frequency'], cltv['monetary'])
cltv["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv['frequency'],
                                                                             cltv['monetary'])
           # 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz. (cltv kullandığım için cltv_6_ olarak yapıcam)
cltv["cltv"] = ggf.customer_lifetime_value(bgf,
                                   cltv['frequency'],
                                   cltv['recency'],
                                   cltv['T'],
                                   cltv['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
                # b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv["cltv"].sort_values(ascending=False)[:20].reset_index(drop=True)
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
           # 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.



cltv["cltv_segment"] = pd.qcut(cltv["cltv"], 4, labels=["D", "C", "B", "A"])
cltv.head()
           # 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz
cltv.groupby("cltv_segment").agg(
    {"count", "mean", "sum"})
#6 ayda A segmentimden ortalama 372.6468 getiri bekliyorum bana ortalma 238 birim ortalma kar bırakabilir   6 aylık satışlara bakacak olur isek
# toplam 7890 adet satış bekleniyor bu rakamları referans alıp hazırlıklarımıza göz atalım arkadaşlar
"""
Ekip arkadaşlarım, elimizde oldukça net bir tablo var. 🔍 A segmenti müşterilerimiz, bizim en değerli müşterilerimiz 
arasında yer alıyor. Önümüzdeki 6 ay içinde bu segmentten ortalama 372.65 birim gelir bekliyoruz. Ancak işin güzel 
tarafı, bu sadece ciro değil! Bu müşteriler ortalama 238.76 birim kâr bırakacaklar. Yani, onları elde tutmak ve alışveriş 
sıklığını artırmak bizim için kritik bir kazanç kapısı.
"""

"""
B segmenti için 6 ayda 167.83 birim kar ve 200.47 birim getiri bekleniyor. Bu segmentin satışları 6035.8803 adet olarak tahmin ediliyor. C segmentiyle 
B segmentinin kar marjı arasında 30 birim fark var ve getirileri c segmenti için 660935 birim iken 836780 birim diğer segmentlere kıyasla daha az bir fark var
Bunu gözden kaçırmayalım ve C segmenti için de ipleri biraz sıkı tutalım arkadaşlar
"""
# BONUS: Tüm süreci fonksiyonlaştırınız.

def cltv_prediction(df,outlier_chart=False,segment_describe=False):
    """

    :param df: dataframe tanımlayınız
    :param outlier_chart: aykırı değerler analiz eilirken grafikleri takip etmenizi sağlar
    :param segment_describe: cltv segmentlerine göre betimsel istatistiklerini verir
    :return:
    """
    def outlier_thresholds(dataframe, variable):
        quartile1 = dataframe[variable].quantile(0.01)
        quartile3 = dataframe[variable].quantile(0.99)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit


    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)

        if (dataframe[variable] > up_limit).any():
            print("***************  OUTLİER !!  ****************")
            print(f"Üst eşik değerini aşan değişken: {variable} \nBelirlenen üst eşik değeri: {up_limit}")

        if (dataframe[variable] < low_limit).any():
            print("***************  OUTLİER !!  ****************")
            print(f"Alt eşik değerinin altında kalan değişken: {variable} \nBelirlenen alt eşik değeri: {low_limit}")

        dataframe.loc[dataframe[variable] > up_limit, variable] = round(up_limit)
        #dataframe.loc[dataframe[variable] < low_limit, variable] = round(low_limit)

    if outlier_chart:
        for col in df.columns:
            if df[col].dtypes in ["float64"]:
                plt.boxplot(df[col])
                plt.title(col)
                plt.show(block=True)


    for col in df.columns:
        if df[col].dtypes in ["float64"]:
            replace_with_thresholds(df, col)


    df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

    for col in df.columns:
        if "date" in col.lower():
            df[col] = df[col].apply(pd.to_datetime)  # böylede olur df[col].astype("datetime64[ns]")


    today = df["last_order_date"].max() + pd.Timedelta(days=2)

    df["recency"] = round(((df["last_order_date"] - df["first_order_date"]).dt.days) / 7)
    df["T"] = round(((today - df["first_order_date"]).dt.days) / 7)

    cltv = df[["recency", "T", "order_num_total", "customer_value_total"]]
    cltv.columns = ['recency', 'T', 'frequency', 'monetary']

    cltv = cltv[cltv["frequency"] > 1]
    cltv["monetary"] = cltv["monetary"] / cltv["frequency"]

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv['frequency'],
            cltv['recency'],
            cltv['T'])

    cltv["exp_sales_6_month"] = bgf.predict(4 * 6,
                                            cltv['frequency'],
                                            cltv['recency'],
                                            cltv['T'])

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv['frequency'], cltv['monetary'])
    cltv["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv['frequency'],
                                                                              cltv['monetary'])

    cltv["cltv"] = ggf.customer_lifetime_value(bgf,
                                               cltv['frequency'],
                                               cltv['recency'],
                                               cltv['T'],
                                               cltv['monetary'],
                                               time=6,  # 6 aylık
                                               freq="W",  # T'nin frekans bilgisi.
                                               discount_rate=0.01)

    cltv["cltv_segment"] = pd.qcut(cltv["cltv"], 4, labels=["D", "C", "B", "A"])

    if segment_describe:
        comment = cltv.groupby("cltv_segment").agg(
        {"count", "mean", "sum"})
        return cltv , comment

    return cltv


df = df_.copy()
df,comment = cltv_prediction(df,segment_describe=True)
