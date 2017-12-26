# VGG16_DogCat_Classification
### 개요:

 * Windows 환경에서 VGG16모델 적용.
 * Data 수집, traing과정, 결과 설명.
 * Tensorflow로 활용
 * 개25종 + 고양이12종 = 37 calss를 분류.



### VGG 모델
 * [VGG16](https://github.com/machrisaa/tensorflow-vgg)을 다운받아, 필요한 곳을 수정함.
 * VGG모델은 1000개의 [ImageNet](http://www.image-net.org/) 1,000개의 category를 분류하는 학습된 weight가 공개되어 있다.
 * [training된 weight]( https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM)를 다운받을 수 있다.


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
