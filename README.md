# NYCU-Machine-Learning
國立陽明交通大學 113學年度上學期 資科工碩 授課教師：洪瑞鴻、邱維辰

HW1-HW4為洪瑞鴻教授授課部分，每次作業會有demo，助教會問一些和作業有關的問題，包含程式碼的寫法和上課提到的內容，比如HW1可能會問道"為什麼加入regularized term可以解決overfitting?"一個比較合適的回答方式可以參考筆記lesson01中的敘述，當overfitting發生時，高次方項的係數絕對值通常會暴增，因此當我們在計算loss時將係數絕對值納入考量後可以遏止overfitting的狀況。  
答錯或回答時卡詞也不用太擔心，demo的目的主要是助教在確認大家的作業是不是自己寫的，別太誇張的話基本上demo的部分不會被扣分。  

HW5-HW7為邱維辰教授授課部分，不用demo，但每次作業需要繳交report。  
不同於前四次作業會用到的數學公式推導都有教過，後三次作業有些內容甚至在課堂上完全沒提，比如HW7的Kernel-LDA。  
由於這部分的作業分數有一半以上基於report的撰寫，會需要花更多的時間，建議要提早開始寫作業。  
  
  
以下為其他注意事項:  
  
HW1
1. 一些矩陣運算操作不能call library(比如矩陣的乘法、轉置、找反矩陣等)，須特別注意spec的要求；之後幾次作業應該就不用自己寫矩陣運算操作了。  
2. Steepest descent method在加入懲罰項後要計算gradient時，因為是l1-norm的關係，在x=0時不可微，可用sign fuction或epsilon appoximation等方式表達，殊途同歸，最後算出的loss應該都非常接近。  

HW2
1. 在spec中有寫道"Print out the the posterior (in log scale to avoid underflow)"，須注意由於機率的數值範圍為0\~1，取log後剛好會都是負數，在最後normalization時才會負負得正，原先有最高posterior probability的類別在normalization後數值反而會是最低的，所以在程式碼中才會寫成"predicted_label = np.argmin(log_posteriors_normalized)"；但是，在continuous mode時([Gaussian Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes))，實際上是用機率密度函數的表示方式來計算posterior probability，而機率密度函數的數值範圍並不一定會是0\~1，這與剛才的假設不符，會導致有些極端情況算錯；不過在詢問助教後他說可以不用特別解決這個問題，希望之後修課的同學可以更新正確的版本XD  
2. 在某些pixel上可能會出現variance=0的狀況，影響到後續計算，需要額外設定一個數值以避免分母為零。直覺上可能會設定一個接近零的常數，但是在高斯分布中，variance愈小，其機率密度函數的最大值愈大，可能會超過1而導致前面提過的邏輯錯誤；把variance設得太大，又會使整個高斯分布的形狀會愈趨扁平而接近於均勻分布。可以自行嘗試選擇最合適的設定。  

HW4
1. EM演算法的分群結果會受初始值影響，不同的初始化方法會導致最後收斂到不同的分群結果。若要檢查自己的EM演算法實現流程是否寫對，可以使用strong prior(直接記錄在每個pixel上，不同類別的圖像出現'1'的總次數)來幫助驗證其餘程式碼的正確性，針對這種初始化方法，error rate大約是0.31。但前述的初始化方法相當於是在訓練時就使用了類別資訊，不符合作業要求，因此還是必須另外想出一種合適的初始化方法。一般來說，隨機初始化的error rate範圍大約落在0.4~0.6之間。