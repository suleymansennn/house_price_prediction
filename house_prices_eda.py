import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import *

df = pd.read_csv("datasets/train.csv")

missing_values_analysis(df)
""""
PoolQC, MiscFeature, Alley, Fence, FireplaceQu değişkenlerinde çok sayıda eksik gözlem var bu yüzden veri setinden
çıkartılabilir.
"""



selected_num_cols = ["BsmtFinSF1", "GrLivArea", "LotFrontage", "LotArea", "1stFlrSF", "BsmtUnfSF",
                     "GarageArea", "2ndFlrSF", "SalePrice"]
selected_cat_cols = ["Neighborhood", "OverallQual", "BsmtExposure", "KitchenQual", "SaleType",
                     "SaleCondition", "Functional", "Condition1", "ExterQual"]

for col in selected_num_cols:
    num_summary(df, col, "SalePrice")

"""
BsmtFinSF1: Tip 1 bitmiş alanın metre karesi
    - Hafif sağa çarpık bir dağılım var
    - 0 olan gözlemler bodrumun olmadığını gösteriyor.
    - Satış fiyatı ile aralarında düşük bir ilişki var.
    - Aykırı değer yakalanabilir.
GrLivArea: Zemin oturma alanı metre karesi
    - Aykırı değerler var. Bu yüzden dağılım hafif sağa çarpık.
    - Satış fiyatı ile aralarında güçlü bir ilişki var.
LotFrontage: Evin cadde ile doğrudan bağlantı uzunluğu
    - Aykırı değerler var
    - Satış fiyatı ile aralarında düşük bir ilişki var.
    - Gözlemlerin çoğu 50-100 arasında toplanmış
LotArea: Arsa büyüklüğü
    - Aykırı değerler var
    - Sağa çarpık bir dağılıma sahip
    - Satış fiyatı ile aralarında düşük bir ilişki var.
 1stFlrSF: Birinci katın metre kare alanı
    - Aykırı değer yakalanbilir.
    - Normal dağılıma yakın bir dağılıma sahip
    - Satış fiyatı ile aralarında iyi bir ilişki var.
BsmtUnfSF: Bodrumun bitmemiş alanın metre karesi
    - Sağa çarpık bir dağılım var.
    - Aykırı değer olabilir.
    - 0 olan gözlemler bodrumun olmadığını gösterir.
    - Satış fiyatı ile aralarında düşük bir ilişki var.
GarageArea: Garaj metre kare alanı
    - 0 olan gözlemler garajın olmadığını gösterir.
    - Aykırı değerler olabilir.
    - Satış fiyatı ile aralında güçlü bir ilişki var.
2ndFlrSF: 2. katın metre kare alanı
    - 0 olan gözlemler 2. katın olmadığını belirtir.
    - 0 olan gözlemler haricinde normal dağılıma yakın
    -  Satış fiyatı ile aralında güçlü bir ilişki var.
SalePrice: Satış fiyatı (Bağımlı Değişken)
    - Sağa çarpık bir dağılıma sahip
    - Aykırı değerler olabilir.
    - Gözlemler genellikle 100.000 ile 200.000 arasında
"""

neigh_map = {'MeadowV': 1, 'IDOTRR': 1, 'BrDale': 1, 'BrkSide': 2, 'OldTown': 2, 'Edwards': 2,
             'Sawyer': 3, 'Blueste': 3, 'SWISU': 3, 'NPkVill': 3, 'NAmes': 3, 'Mitchel': 4,
             'SawyerW': 5, 'NWAmes': 5, 'Gilbert': 5, 'Blmngtn': 5, 'CollgCr': 5,
             'ClearCr': 6, 'Crawfor': 6, 'Veenker': 7, 'Somerst': 7, 'Timber': 8,
             'StoneBr': 9, 'NridgHt': 10, 'NoRidge': 10}

df['Neighborhood'] = df['Neighborhood'].map(neigh_map).astype('int')

for col in selected_cat_cols:
    cat_plots(df, col, "SalePrice")

"""
Neighborhood: Ames şehir sınırları içindeki fiziksel konumu
    - Encoding işleminden sonra bağımlı değişken açısından güzel bir açıklayıcılığa sahip.
    - Okulla ilgili bir değişken türetilebilir.
    - 9 sınıfı rare encoding'e takılabilir.
OverallQual: Genel malzeme kalitesi
    - Bağımlı değişkeni açıklama konusunda iyi.
    - 1, 2, sınıfları rare encoding'e takılacaktır.
BsmtExposure: Bahçe seviyesindeki duvarları temsil eder
    - Evlerin büyük bir bölümünde yok.
    - Bağımlı değişkeni açıklamak için fena değil.
    - Duvarı olmayan evlerin ortalama fiyatı daha az.
KitchenQual: Mutfak kalitesi (Excellent, Good, Average, Fair, Poor)
    - Sınıflar bağımlı değişken açısından iyi ayrılmış.
    - Sınıflar içinde aykırılıklar yok.
    - Belki bu ayrımı bir model ile yapılmıştır.
SaleType: Satış TÜrü
    - Bir çok sınıf rare encoding'e takılacaktır.
    - Gözlem sayısı az olan sınıfları göz ardı edersek, bağımlı değişken açısından güzel ayrılmış.
SaleCondition: Satış Durumu
    - Rare encoding'e takılacak sınıflar var.
    - Gözlemlerin büyük çoğunluğu normal satış
    - Değişkenin sınıfları bağımlı değişken açısından güzel ayrılmış.
Functional: Evin işlevselliği
    - Sınıflar rare encoding'e takılacaktır.
    - 93%'ü tek bir sınıfa ait. Veri setinden çıkartılabilir.
Condition1: Caddeye veya tren yoluna yakınlık.
    - Bir çok sınıfının oranı az.
    - Gözlemi az olan sınıfların aralında satış fiyatı olarak farklılıklar var. Bu yüzden bu gözlemleri manuel encode
    edebilirim.
ExterQual: Dış malzeme kalitesi.
    - Sınıflar arasında bağımlı değişken açısından iyi ayrılmış.
"""

corr = df[selected_num_cols].corr()
corr_matrix = corr.abs()
msk = np.triu(corr)
sns.heatmap(corr, mask=msk, annot=True, linecolor="black", cmap=cat_feat_colors)
plt.yticks(rotation=0, size=10)
plt.xticks(rotation=75, size=10)
plt.title('\nCorrelation Map\n', size=15)
plt.show(block=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

rare_analyser(df, "SalePrice", cat_cols)
"""
Street, Alley, Utilities, RoofMatl, Heating, Condition2 değişkenleri veri setinden çıkartılabilir.
"""
