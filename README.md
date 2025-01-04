# NYCU-Machine-Learning
NYCU 2024 Spring Machine Learning / 洪瑞鴻教授、邱維辰教授

HW1-HW4為洪瑞鴻教授授課部分，每次作業會有demo，助教會問一些和作業有關的問題，包含程式碼的寫法和上課提到的內容
比如HW1可能會問到"為什麼加入regularized term可以解決overfitting?"一個比較合適的回答方式可以參考筆記lesson01中的敘述，當overfitting發生時，高次方項的係數絕對值容易暴增，因此當我們在計算loss時將係數絕對值納入考量後可以遏止overfitting的狀況
答錯或回答時卡詞也不用太擔心，demo的目的主要是助教在確認大家的作業是不是自己寫的，別太誇張的話基本上demo的部分不會被扣分

HW5-HW7為邱維辰教授授課部分，不用demo，但每次作業需要繳交report
不同於前四次作業會用到的數學公式推導上課都有教過，後三次作業有些內容甚至在課堂上完全沒提，比如HW7的Kernel-LDA
加上這部分的作業分數有一半以上基於report的撰寫，會需要花更多的時間，建議要更早開始寫作業


其他注意事項:
HW1: 
1. 一些矩陣運算操作不能call library(比如矩陣的乘法、轉置、找反矩陣等)，須特別注意spec的要求；之後幾次作業應該就不用自己寫矩陣運算操作了
2. Steepest descent method在加入懲罰項後要計算gradient時，因為是l1-norm的關係，在x=0時不可微，可用sign fuction或epsilon appoximation等方式表達，殊途同歸，最後算出的loss應該都非常接近

HW2:
1. 在spec中有寫道"Print out the the posterior (in log scale to avoid underflow)"，須注意由於機率的數值範圍為0~1，取log後剛好會都是負數，在最後normalization時才會負負得正，因此，原先有最高posterior probability的類別在normalization後數值反而會是最低的，所以在程式碼中才會寫成"predicted_label = np.argmin(log_posteriors_normalized)"；但是，在continuous mode時(Gaussian Naive Bayes [text](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes))實際上是用機率密度函數的表示方式來計算posterior probability，而機率密度函數的數值並不一定會是0~1，與剛才的假設不符，不過在詢問助教後他說不用特別解決這個問題