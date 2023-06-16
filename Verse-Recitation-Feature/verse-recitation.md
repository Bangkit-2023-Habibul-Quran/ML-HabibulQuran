## Model Introduction

The model is a fine-tuned version of the Wav2Vec2 XLS-R with 300 m parameter model designed for Quran verse recitation. The base model, Wav2Vec2 XLSR, is a state-of-the-art automatic speech recognition (ASR) model that has been pre-trained on a large amount of multilingual audio data. We load the model with transformer huggingface module to use the pre-trained Wav2Vec2 model.

## Dataset

For the fine-tuning process, the Wav2Vec2 XLSR model was trained on a dataset specifically created for Quran recitation. 
The dataset used for fine-tuning the model is called Ar-DAD (Arabic Diversified Audio Dataset), for the detail : https://data.mendeley.com/datasets/3kndp5vs6b/3

This dataset contains audio recordings of Quranic verses along with their corresponding transcriptions. By training the model on this dataset, it learns to recognize and generate accurate transcriptions for Quranic verses.

## Project Steps

The project follows these steps:

1. Prepare Data, Tokenizer, Feature Extractor
2. Preprocess Data : load and resample the audio data, extract the input_values, encode the transcriptions to label ids
3. Training : Set up trainer, Feed the preprocessed data, along with the tokenized text and extracted audio features
4. Evaluation : Measure metrics word error rate to assess how well the model can transcribe Quranic verses.
5. Convert Model to HDF5 : To ensure compatibility with TensorFlow, the pretrained model's output, pytorch_model.bin, can be converted to HDF5 format. 
6. Test ASR model prediction

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-4
- train_batch_size: 50
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- optimizer: Adamw_torch
- lr_scheduler_type: linear
- num_epochs: 25 (but 20 epoch succes 5 fail to run and save)
- mixed_precision_training: Native AMP

## Training and Evaluation Results

The following table shows the training and evaluation results for the model:

| Epoch | Training Loss | Validation Loss | WER (Word Error Rate) |
|-------|---------------|-----------------|----------------------|
| 1     | 3.822900      | 2.036937        | 1.047724             |
| 2     | 1.011200      | 0.161243        | 0.500000             |
| 3     | 0.225500      | 0.093763        | 0.366373             |
| 4     | 0.194500      | 0.092610        | 0.358297             |
| 5     | 0.132900      | 0.081145        | 0.342878             |
| 6     | 0.109500      | 0.079725        | 0.333333             |
| 7     | 0.103400      | 0.085686        | 0.331865             |
| 8     | 0.098700      | 0.079209        | 0.329662             |
| 9     | 0.106800      | 0.081487        | 0.327460             |
| 10    | 0.087500      | 0.081098        | 0.325257             |
| 11    | 0.075700      | 0.070294        | 0.315712             |
| 12    | 0.074200      | 0.074002        | 0.317915             |
| 13    | 0.072100      | 0.073507        | 0.315712             |
| 14    | 0.070300      | 0.073163        | 0.316446             |
| 15    | 0.062900      | 0.075280        | 0.314978             |
| 16    | 0.067300      | 0.078965        | 0.312775             |
| 17    | 0.082200      | 0.074298        | 0.312041             |
| 18    | 0.061900      | 0.073618        | 0.315712             |
| 19    | 0.060100      | 0.074851        | 0.314244             |
| 20    | 0.059200      | 0.077483        | 0.312775             |

# Model Summary

This model has undergone multiple rounds of fine-tuning using the Wave2Vec2 model. This iterative fine-tuning process was repeated several times to progressively enhance the model's performance and capabilities. Finally, the model presented here is the result of the fine-tuning process based on zarko1231/model-baru-collab using the underlying architecture from facebook/wav2vec2-xls-r-300m.

It achieves the following results on the evaluation set:
- Training Loss: 0.059200
- Validation Loss : 0.077483
- Wer: 0.312775

### Framework versions

- Transformers 4.28.0
- Pytorch 2.0.1+cu118
- Datasets 2.12.0
- Tokenizers 0.13.3
