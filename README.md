# NYCU-Machine-Learning
國立陽明交通大學 113學年度上學期 資科工碩 授課教師：洪瑞鴻、邱維辰

HW1-HW4為洪瑞鴻教授授課部分，每次作業會有demo，助教會問一些和作業有關的問題，包含程式碼的寫法和上課內容，比如HW1可能會問道"為什麼加入regularized term可以解決overfitting?"一個比較合適的回答方式可以參考筆記lesson01中的敘述，當overfitting發生時，高次方項的係數絕對值通常會很大，因此當我們在計算loss時將係數納入考量後可以遏止overfitting的狀況。  
答錯或回答時卡詞也不用太擔心，demo的目的主要是助教在確認大家的作業是不是自己寫的，別太誇張的話基本上demo的部分不會被扣分。  

HW5-HW7為邱維辰教授授課部分，不用demo，但每次作業需要繳交report。  
不同於前四次作業會用到的公式推導都有教過，後三次作業有些內容甚至在課堂上完全沒提，比如HW7的Kernel-LDA。  
由於這部分的作業分數有一半以上基於report的撰寫，會需要花更多的時間，建議要提早開始寫作業。  
  
  
以下為作業注意事項:  
  
HW1
1. 一些矩陣運算操作不能call library(比如矩陣的乘法、轉置、找反矩陣等)，須特別注意spec的要求；之後的作業應該就不用自己寫矩陣運算了。  
2. Steepest descent method在加入懲罰項後要計算gradient時，因為是l1-norm的關係，在x=0時不可微，可用sign fuction或epsilon appoximation等方式表達，殊途同歸，最後算出的loss應該都非常接近。  

HW2
1. 在spec中有寫道"Print out the the posterior (in log scale to avoid underflow)"，須注意由於機率的數值範圍為0\~1，取log後剛好會都是負數，在最後normalization時才會負負得正，原先有最高posterior probability的類別在normalization後數值反而會是最低的，所以在程式碼中才會寫成"predicted_label = np.argmin(log_posteriors_normalized)"；但是，在continuous mode時([Gaussian Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes))，實際上是用機率密度函數來計算posterior probability，而機率密度函數的數值範圍並不一定會是0\~1，這與剛才的假設不符，會導致有些極端情況出錯；不過在詢問助教後他說可以不用特別解決這個問題，希望之後修課的同學可以更新修正的版本XD  
2. 在某些pixel上可能會出現variance=0的狀況，影響到後續計算，需要額外設定一個數值以避免分母為零。直覺上可能會設定一個接近零的常數，但是在高斯分布中，variance愈小，其機率密度函數的峰值愈大，可能會超過1而導致前面提過的邏輯錯誤；把variance設得太大，又會使整個高斯分布的形狀會愈趨扁平而接近於均勻分布。可以自行嘗試選擇最合適的設定。  

HW3
1. 從spec敘述來看，input的a應為高斯分布的variance，但是在上課推導和筆記的內容中一般將a表示為1/variance，需要注意在程式碼中變數和符號等細節差異。

HW4
1. EM演算法的分群結果會受初始值影響，不同的初始化方法會導致最後收斂到不同的分群結果。若要檢查自己的EM演算法流程是否寫對，可以使用strong prior(直接記錄在每個pixel上，不同類別的圖像出現"1"的總次數)來幫助驗證其餘程式碼的正確性，針對這種初始化方法，error rate大約是0.31。但前述的初始化方法相當於是在訓練時就使用了類別資訊，不符合作業要求，因此還是必須另外想出一種合適的初始化方法。一般來說，隨機初始化的error rate範圍大約落在0.4~0.6之間。  

HW5
1. 最新版本的libsvm和scipy套件似乎會有衝突(參考我的report中"Observations and Discussion"章節所提及)，可以安裝舊版套件以正確執行，套件版本在report的第九頁有紀錄。  

HW6
1. 使用kernel method的核心思想是**避免直接計算(高維)feature space的feature representation**，而根據上課所推公式執行kernel kmeans演算法的話，各群在feature space中的質心位置應該會自動隱式(implicitly)變更，不需要經過計算或額外更新座標。我在參考github上其他人的寫法中常看到註解含"E step..., M step..."，會需要分成這兩個步驟是因為在一般空間中，kmeans演算法需要更新各群質心的位置(M step)，但在kernel kmeans演算法中就不應出現這種寫法。需特別注意**kernel matrix儲存的是兩點之間的similarity，而不直接儲存座標或距離等資訊**，因此諸如在一般特徵空間中，收集某個群內所有成員的座標，取mean計算質心位置之類的操作，並不能直接透過對kernel取mean來實現。  

HW7
1. 執行PCA和LDA演算法過程中會需要建立covariance matrix，其shape為(num_features, num_features)，而原始圖像大小為(195, 231)，也就是說num_features=195x231=45045。PCA可以透過公式化簡(見report)來避免用這麼龐大的矩陣進行運算，但LDA因為涉及計算反矩陣，不容易化簡，我最後採取的作法是先將圖片reshape成(65, 77)再做後續的運算。參考github上其他人的作業時常看到"先用PCA對圖片降維，然後再進行LDA"的流程，我在網路上有找到類似作法的reference([連結1](https://reetuhooda.github.io/webpage/files/Milestone2.pdf))([連結2](https://www.sciencedirect.com/science/article/pii/S0031320302000481?ref=pdf_download&fr=RR-2&rr=8fd137831c448270))，但這並非上課所教的一般LDA流程，需特別注意。  
2. 理論上，用LDA演算法只能將data points投影到(num_class - 1)維空間，而本次作業的人臉類別為15種，故應該只有14張fisherfaces；我其實不確定為什麼輸出的25張臉中後面的11張還是看得出人臉輪廓(我原本預期會是雜訊)XD  
助教們有輸出和我差不多的，也有只剩14張臉的，我們討論後還是不確定正解應該長怎樣，這個bug(?)就交給之後修課的同學修正了。  
3. 上課沒有教Kernel-LDA，我的作法是把原公式改寫成feature representation後再展開成dot product的形式，然後合併成kernel trick的表示形式(見report)。但最後測試的performance感覺沒那麼合理(Accuracy - LDA: 9x% - > Kernel LDA: 8x%)，也許作業中仍存在編寫邏輯的錯誤，請特別注意。([其他推導公式的tutorial](https://www.youtube.com/watch?v=WnzrzEXTyIQ))  
4. Kernel-PCA和Kernel-LDA的計算過程中不會用到covariance matrix，因此應該無法像task 1一樣show eigenfaces/fisherfaces，若有看到其他人的寫法可以印出Kernel-PCA/Kernel-LDA的eigenfaces/fisherfaces，請特別檢查這部分的程式碼，很可能有錯。  
