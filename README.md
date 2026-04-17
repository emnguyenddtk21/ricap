# A PyTorch implementation of RICAP

Repository này huấn luyện WideResNet trên CIFAR-10/CIFAR-100 với:

- baseline chuẩn
- RICAP (Random Image Cropping and Patching)
- Mixup
- Random Erasing

Logic RICAP trong code bám theo paper: lấy 4 ảnh, crop theo biên lấy mẫu từ `Beta(beta, beta)`, patch lại thành 1 ảnh mới, rồi trộn loss/label theo tỉ lệ diện tích từng mảnh.

## Môi trường

- Python 3.10+
- PyTorch 2.2+
- torchvision 0.17+

Cài nhanh:

```bash
pip install -r requirements.txt
```

## Train

### CIFAR-10 baseline

```bash
python train.py --dataset cifar10
```

### CIFAR-10 + RICAP

```bash
python train.py --dataset cifar10 --ricap
```

### CIFAR-10 + Mixup

```bash
python train.py --dataset cifar10 --mixup
```

### CIFAR-10 + Random Erasing

```bash
python train.py --dataset cifar10 --random-erase
```

## Các tuỳ chọn hữu ích

```bash
python train.py \
  --dataset cifar10 \
  --ricap \
  --batch-size 128 \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers \
  --amp
```

Một số thay đổi đã được cập nhật để phù hợp PyTorch/Python hiện tại:

- bỏ `.cuda()` hardcode, thay bằng `device`
- bỏ API cũ như `DataFrame.append()` và `scheduler.get_lr()`
- thêm checkpoint `latest` và `best`
- dùng `torchvision.transforms.RandomErasing` thay cho bản custom lỗi xác suất
- thêm `num_workers`, `pin_memory`, `persistent_workers`, `amp`, `seed`, `batch_size`, `data_dir`, `output_dir`

## Colab

Notebook chạy trên Colab nằm ở:

- `ricap_colab.ipynb`

Notebook này cài dependency, cấu hình runtime, và gọi trực tiếp `train.py`.
