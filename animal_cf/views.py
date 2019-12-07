from django.shortcuts import render, redirect, reverse
from django.conf import settings as settings
from .forms import AnimalImageForm
from .models import AnimalImage
import numpy as np
import cv2
import os
from .googlenet import GoogleNet
import chainer
import chainer.links as L
import chainer.functions as F

model = L.Classifier(GoogleNet())
base_dir = settings.BASE_DIR
npz_file = (base_dir +'/model_gnet_transfer.npz')
chainer.serializers.load_npz(npz_file, model)

# Create your views here.

def index(request):
    if request.method == "GET":
        context = {
            'message':'動物の画像を選択してください',
            'form':AnimalImageForm,
        }
        return render(request, 'animal_cf/index.html', context)
    
    elif request.method == 'POST':
        print(request.POST)
        print(request.FILES)
        # print(request.IMAGES)
        form = AnimalImageForm(request.POST, request.FILES)
        if not form.is_valid():
            # validation失敗
            context = {
                'message': 'ファイルがおかしいです',
                'form': AnimalImageForm
            }
            return render(request, 'animal_cf/index.html', context)
        else:
            # 保存
            form.save()
            # 最新の画像のファイルパスを取得
            posted_img = AnimalImage.objects.order_by('-id')[0]
            posted_img_path = posted_img.animal_image.path
            # 画像の変更
            img = imread(posted_img_path)
            img = img_translate(img)
            label_num = img_estimater(img, model)
            print(label_num)
            animal_name = animal_number_to_name(label_num)

            context = {
                'message': 'この動物は',
                'estimate_result': animal_name,
            }

            return render(request, 'animal_cf/result.html', context)

# WindowsでOpenCVを使う場合、ファイルパスに日本語が入っているとうまくいかない
# 参考URL:https://qiita.com/SKYS/items/cbde3775e2143cad7455
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.float32):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def img_translate(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = img.astype('f').transpose(2,0,1)
    return img

def img_estimater(img, model):
    y = model.predictor(np.array([img])) # imgを3次元形式に変換
    y = y.array # yをchainer形式からnumpy形式に変換
    label_num = np.argmax(y, axis=1)[0]
    return label_num

def animal_number_to_name(num):
    animal_list = [
        'アザラシ', 
        'アシカ', 
        'カバ', 
        'カワウソ', 
        'カンガルー', 
        'キリン', 
        'ゴリラ', 
        'サイ', 
        'サル', 
        'シマウマ', 
        'ゾウ', 
        'トラ', 
        'パンダ', 
        'ハシビロコウ', 
        'フラミンゴ', 
        'ペンギン', 
        'ホッキョクグマ', 
        'ライオン', 
        'レッサーパンダ', 
        'ワニ'
        ]
    return animal_list[num]