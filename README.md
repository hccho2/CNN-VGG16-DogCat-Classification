# VGG16_DogCat_Classification
### 개요:

 * Windows 환경에서 VGG16모델 적용.
 * Data 수집, traing과정, 결과 설명.
 * Tensorflow로 활용
 * 개25종 + 고양이12종 = 37 calss를 분류.



### VGG 모델
 * [VGG16](https://github.com/machrisaa/tensorflow-vgg)을 다운받아, 필요한 곳을 수정함.
 * VGG모델은 1000개의 [ImageNet](http://www.image-net.org/) 1,000개의 category를 분류하는 학습된 weight가 공개되어 있다.
 * [training된 weight(vgg16.npy)]( https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM)를 다운받을 수 있다.


### 새로운 data 확보 과정
 * 첫번째 시도
	+ [64x64](https://github.com/MyHumbleSelf/cnn_assignments/tree/master/assignment3/cs231n/datasets) 이미지를 다운받음.
	+ 100개의 category로 구성. 각 category는 500개의 64x64x3 크기의 image file(jpeg)로 되어 있다. 
	모두 5만개의 파일들인데, 파일 사이즈는 약 100M정도 이다. 
	+ 파일 전체를 로드하는데는 시간이 지남에 따라, 같은 수의 이미지 파일을 로드하는데도 점점 느려짐.
	+ 그래서 10또는 20개의 npz파일로 나누어 저장함. 10개의 npz로 나눌 경우, 각각의 npz파일 크기는 150M정도.
	+ 나누어지 npz파일을 np.concatenate로 다시 합치면, 약 1.5G. numpy array data를 float으로 하지 않고, np.int형으로 저장하면 파일 사이즈는 절반으로 줄어든다. pickle파일로 저장할 경우, 파일 사이즈는 훨씬 커짐.
	+ 1.5G의 npz파일을 메모리로 올리면, 4.8G정도가 된다.
	+ 합쳐진 image data전체는 많은 메모리를 차지하기 때문에, 메모리가 충분하지 못할 경우를 대비하여 image의 순서를 shuffle하여 10개 정도로 나누어 저장.
	+ 이렇게 작업을 한 이후, 뒤늦게 VGG모델은 244x224 크기의 이미지를 사용하는 모델임을 다시 알게됨. 다운 받은 64x64를 resize해서 더 크게 만들어야 함.
	``` js
			resized_img = skimage.transform(img,(224,224),preserve_range = True) 
	```
	+ preserve_range = False 이면, 0~1값으로 바뀜. True로 옵션을 주면, 정수 형태지만 type은 np.float. 실행 속도도 느림.
	+ 어째든 해상도가 낮아, VGG모델을 수정하여 저해상도에 맞게 적용하면 되겠지만, VGG를 바로 적용해 보고 싶어서 다른 Dataset을 찾아보기로 함. 

 * 두번째 시도
	+ [DeepLearning Dataset](http://deeplearning.net/datasets/)에서 Caltec Image를 다운받음. Category101, Category256이 있음.
	+ Caltech256 image: category가 256개(256_ObjectCategories.tar-1.15G(압축 풀면, 257개 category}). image들의 size는 동일하지 않고 다양함(eg. 499 x 278, 200 x 150, 1024 x 470, ...). 전체 image 개수는 30,608개.
	+ Caltech101 image: category가 101개(128M. 이것도 실제 102개 category). category별 image 개수 차이가 많이 남(eg. car-side는 모두 흑백 이미지. motorbikes는 798개). 전체 image 개수는 9,145개. int형의 npz 파일로 저장하면 1.12G(흑백 이미지를 제외하면 8,733개)
	+ Deep Learning을 수행하기 위해서는 category당 1,000여개의 이미지가 있어야 하는데, category당 80~90개는 너무 적다.
	+ category별로 보면, 40개 category는 50장 이하, 49개 category는 50~100장, 그 외 category는 100장, 200장, 400장, 800장 등. 
	+ category별 이미지 개수의 편차가 심해, VGG training data로 사용하기에는 부족함.
	
 * 세번째 시도
	+ [Vgg Dataset](http://www.robots.ox.ac.uk/~vgg/data/)에서도 여러가지 data를 구할 수 있다.
	+ Pet Dataset: 25 category의 개 사진과 12 category의 고양이 사진이 7,390장 있다. category별로 200개 정도로 균일하다. 
	+ 'Egyptian)_Mau_129.jpg', 'staffordshire_bull_terrier_2.jpg', 'staffordshire_bull_terrier_22.jpg' 3개의 이미지는 흑백이라 제외. 남은 data는 총 7,387개.
	+ 위에서 마찬가지로 많은 이미지 파일을 단일 loop로 읽어 들이면, 점점 느려지기 때문에, category 1개 씩 batch로 처리하여 37개의 npz파일을 먼저 생성.
	+ 37개 npz파일을 Memory에 load하고 이미지 순서를 shuffle한 후, 10개의 npz파일로 나누어 저장함. 
	+ 일단, 이렇게 구한 Pet Data로 VGG16 모델을 돌려보기로 하자.

![VGG](./vgg16.png)	
### VGG 모델 설명
 * VGG모델의 자세한 설명은 따로 정리한 [여기](https://drive.google.com/open?id=1jhQejNuZLCRvWQrWhwxaR1qO1CTp-gcn)에서 볼 수 있다.


### Traing 준비
 * category별로 200개 정도의 data밖에 없기 때문에, train data, validation data, test data로 나누지 않고, 모두 train data로 사용.
 * test는 training이 끝나면, 인터넷에서 몇 개 다운받아서 해보기로 함.
 * weight의 초기값을 random으로 주지 않고, 다운 받은 vgg16.npy에 있는 값으로 사용해보자. 이러한 방법을 고상하게 [transfer learning](http://cs231n.github.io/transfer-learning/)이라 한다.
 * 기존에 학습된 VGG모델과 지금 training하려는 것의 결정적인 차이는 분류하고자 하는 카테고리의 개수가 다르다는데 있다.
 * 그래서 마지막 fc8 layer는 training된 weight를 가져오지 않고, random으로 초기화함.
 * 기존 VGG16구현은 category가 1,000개로 고정되어 있어서, 이 부분도 일부 수정함.
 * mini batch size를 얼마로 할까 고민하다, 1,000를 분류할 때, 256개를 mini batch size로 했다는 것을 [보고](http://cs231n.github.io/optimization-1/), 37개 category 이므로, 
 mini batch size = 10으로 결정함.

### Traing Machine
 * 동일한 모델을 cpu만 있는 pc와 gpu가 있는 pc에 각각 돌려보기로 함.
 * cpu only pc: Intel i7-2600(3.4GHz), Memory 8G
 * gpu pc: Xeon I3-1225 V3(3.2GHz), Memory 16G, GPU-GTX970


### Traing 소요 시간
 * cpu only pc: Iteration 1,200번 - 50시간 소요
 * gpu pc: Iteration 100,000번 - 11.23시간 소요
 * 두 pc 모두 cost값은 잘 감소한 것으로 보여짐.
 * train data 전체에 대한 accuracy를 계산하지 못하고, 일단 10개만 sample해서 뽑아 보았을 때는 1.0이 나옴.
 * iteration 별 cost값을 저장했어야 하는데, 저장하지 못해 시간 날 때, 다시 돌여야...

### Test결과
 * test_data 디렉토리에 있는 5개 파일을 인터넷에서 다운.
	``` js
	0 correct: Bombay, predict: Bombay ==> True
	1 correct: keeshond, predict: keeshond ==> True
	2 correct: Ragdoll, predict: Ragdoll ==> True
	3 correct: scottish_terrier, predict: scottish_terrier ==> True
	4 correct: english_setter, predict: english_setter ==> True	
	```
 * 일단, 5개의 test data로 평가했을 때는, 결과가 나쁘지 않음.
 * 추후, 더 많은 test data를 확보하여 테스트해야 함.
 
### 코드 실행
 * hccho_VGG_Pet.py ==> train_from_pretrained(): training 과정 구현, weight 값을 npy에 저장함.
 * hccho_VGG_Pet.py ==> predict_from_pretrained(): 저장된 npy파일로 부터 weight 값을 읽어, prediction 수행.
 
