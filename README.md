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
	+ 이렇게 작업한 후, VGG모델은 244x224$ 크기의 이미지를 사용하는 모델임을 다시 알게됨. 다운 받은 64x64를 resize해서 더 크게 만들어야 함.
	``` js
			resized_img = skimage.transform(img,(224,224),preserve_range = True) 
	```
	+ preserve_range = False 이면, 0~1값으로 바뀜. True로 옵션을 주면, 정수 형태지만 type은 np.float. 실행 속도도 느림.
	+ 어째든 해상도가 낮아, VGG모델을 수정하여 저해상도에 맞게 적용하면 되겠지만, VGG를 바로 적용해 보고 싶어서 다른 Dataset을 찾아보기로 함.



