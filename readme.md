# Deep Learning model implementation and optimization

## master 
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)           │ (None, 150, 150, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ resnet152 (Functional)               │ (None, 5, 5, 2048)          │      58,370,944 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling2d             │ (None, 2048)                │               0 │
│ (GlobalAveragePooling2D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 6)                   │          12,294 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
```
 Total params: 58,383,238 (222.71 MB)
 Trainable params: 12,294 (48.02 KB)
 Non-trainable params: 58,370,944 (222.67 MB)

### Results 
```
Epoch 11/20
351/351 ━━━━━━━━━━━━━━━━━━━━ 663s 2s/step - loss: 0.0927 - sparse_categorical_accuracy: 0.9689 
Epoch 12/20
351/351 ━━━━━━━━━━━━━━━━━━━━ 562s 2s/step - loss: 0.0887 - sparse_categorical_accuracy: 0.9702
Epoch 13/20
351/351 ━━━━━━━━━━━━━━━━━━━━ 565s 2s/step - loss: 0.0886 - sparse_categorical_accuracy: 0.9663 
Epoch 14/20
351/351 ━━━━━━━━━━━━━━━━━━━━ 552s 2s/step - loss: 0.0803 - sparse_categorical_accuracy: 0.9739
Epoch 15/20
351/351 ━━━━━━━━━━━━━━━━━━━━ 524s 1s/step - loss: 0.0727 - sparse_categorical_accuracy: 0.9759
Epoch 16/20
351/351 ━━━━━━━━━━━━━━━━━━━━ 550s 2s/step - loss: 0.0730 - sparse_categorical_accuracy: 0.9787
Epoch 17/20
351/351 ━━━━━━━━━━━━━━━━━━━━ 519s 1s/step - loss: 0.0661 - sparse_categorical_accuracy: 0.9819
Epoch 18/20
351/351 ━━━━━━━━━━━━━━━━━━━━ 381s 1s/step - loss: 0.0670 - sparse_categorical_accuracy: 0.9786
Epoch 19/20
351/351 ━━━━━━━━━━━━━━━━━━━━ 385s 1s/step - loss: 0.0597 - sparse_categorical_accuracy: 0.9828
Epoch 20/20
351/351 ━━━━━━━━━━━━━━━━━━━━ 384s 1s/step - loss: 0.0563 - sparse_categorical_accuracy: 0.9831
19/19 ━━━━━━━━━━━━━━━━━━━━ 22s 1s/step - loss: 0.4367 - sparse_categorical_accuracy: 0.9052
Test accuracy with trained teacher model:90.17 %
```

## student
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)           │ (None, 150, 150, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ efficientnetb1 (Functional)          │ (None, 5, 5, 1280)          │       6,575,239 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling2d             │ (None, 1280)                │               0 │
│ (GlobalAveragePooling2D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 1280)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 15)                  │          19,215 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 15)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 6)                   │              96 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
```
 Total params: 6,594,550 (25.16 MB)
 Trainable params: 19,311 (75.43 KB)
 Non-trainable params: 6,575,239 (25.08 MB)

### Results 
```
Epoch 11/20
702/702 ━━━━━━━━━━━━━━━━━━━━ 201s 286ms/step - loss: 0.2303 - sparse_categorical_accuracy: 0.9160
Epoch 12/20
702/702 ━━━━━━━━━━━━━━━━━━━━ 213s 303ms/step - loss: 0.2427 - sparse_categorical_accuracy: 0.9123
Epoch 13/20
702/702 ━━━━━━━━━━━━━━━━━━━━ 204s 291ms/step - loss: 0.2404 - sparse_categorical_accuracy: 0.9147
Epoch 14/20
702/702 ━━━━━━━━━━━━━━━━━━━━ 205s 292ms/step - loss: 0.2261 - sparse_categorical_accuracy: 0.9184
Epoch 15/20
702/702 ━━━━━━━━━━━━━━━━━━━━ 198s 282ms/step - loss: 0.2342 - sparse_categorical_accuracy: 0.9145
Epoch 16/20
702/702 ━━━━━━━━━━━━━━━━━━━━ 206s 294ms/step - loss: 0.2309 - sparse_categorical_accuracy: 0.9168
Epoch 17/20
702/702 ━━━━━━━━━━━━━━━━━━━━ 204s 290ms/step - loss: 0.2228 - sparse_categorical_accuracy: 0.9177
Epoch 18/20
702/702 ━━━━━━━━━━━━━━━━━━━━ 208s 296ms/step - loss: 0.2216 - sparse_categorical_accuracy: 0.9158
Epoch 19/20
702/702 ━━━━━━━━━━━━━━━━━━━━ 207s 294ms/step - loss: 0.2147 - sparse_categorical_accuracy: 0.9187
Epoch 20/20
702/702 ━━━━━━━━━━━━━━━━━━━━ 207s 295ms/step - loss: 0.2259 - sparse_categorical_accuracy: 0.9142
38/38 ━━━━━━━━━━━━━━━━━━━━ 12s 282ms/step - loss: 0.2917 - sparse_categorical_accuracy: 0.8980
Test accuracy with trained teacher model:90.00 %
```