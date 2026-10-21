# SRR

Speech deepfake detection using super-resolution responses and teacher–student distillation.

## 1. Dependencies

Use Python 3.10 with a CUDA-enabled PyTorch/torchaudio installation (tested with PyTorch 2.4). Install FFmpeg and the detector dependencies:

```bash
pip install -r requirements.txt
```

Install [fairseq](https://github.com/facebookresearch/fairseq) following its instructions. Download the [XLS-R 300M checkpoint](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt) from the [official XLS-R release](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr). The detector uses all 24 Transformer layers. The default `--xlsr-output-layer 23` supports standard fairseq; use `-1` only with a fairseq fork that returns all layers when no output layer is specified.

For super-resolution preparation, install [FlashSR](https://github.com/jakeoneijk/FlashSR_Inference) and download its `student_ldm.pth`, `sr_vocoder.pth`, and `vae.pth` using the upstream instructions. FlashSR can run in a separate environment. Its source and weights are not bundled here.

## 2. Prepare speech and super-resolution pairs

Create JSONL manifests with one utterance per line. Labels are `0` for Real and `1` for Fake; utterance IDs must be unique. Paths refer to your own audio files.

```json
{"utterance_id":"example_real","audio_path":"/path/to/real.wav","label":0}
{"utterance_id":"example_fake","audio_path":"/path/to/fake.wav","label":1}
```

Generate aligned 16/48-kHz training pairs in the FlashSR environment:

```bash
python generate_superres.py --flashsr-root /path/to/FlashSR_Inference \
  --weights-dir /path/to/FlashSR_weights --manifest data/train.jsonl \
  --output-dir data/pairs --split train
```

For teacher evaluation, use the same command with the evaluation manifest, a different `--split`, and `--first-only`. Estimate response normalization using training pairs only, in the detector environment:

```bash
python compute_response_stats.py --manifest data/pairs/train.jsonl \
  --output data/response_stats.npz
```

## 3. Train

```bash
GPU_IDS=0,1 bash train.sh --xlsr-checkpoint weights/xlsr2_300m.pt \
  --train-manifest data/pairs/train.jsonl --dev-manifest data/dev.jsonl \
  --response-stats data/response_stats.npz
```

The generic entry point jointly trains the teacher and student with a shared DGSE. Defaults: 50 epochs, AdamW with learning rate `1e-6`, weight decay `0.01`, RawBoost algorithm 4, distillation weight `0.3`, and temperature `2`. Training uses aligned random 64,600-sample crops at 16 kHz and the corresponding 48-kHz crops. Short inputs are zero-padded with valid-length masks. Early stopping uses development loss with patience 10. `BATCH_SIZE` controls the per-device microbatch; `GLOBAL_BATCH` controls gradient accumulation (default 60). Use `--xlsr-checkpointing` to reduce activation memory.

`student.py` implements DGSE, RPM, and RCDR; `teacher.py` adds MSRM and the teacher classifier. `distillation.py` implements prediction, routing, and response distillation. `train.py` also supports `--mode teacher` (requires `--dev-sr-manifest`) and `--mode student`.

Best branch checkpoints contain model tensors only. Local `resume.pt` additionally stores optimizer state and training configuration for resuming with `--resume`.

## 4. Evaluate

Download the checkpoints from [Hugging Face](https://huggingface.co/JunXueTech/SRR):

- `19LA_Teacher.pt`
- `19LA_Student.pt`
- `ASV5_Student.pt`
- `RTCFake_Student.pt`

Student inference uses only the original speech:

```bash
GPU_IDS=0 bash eval_student.sh --xlsr-checkpoint weights/xlsr2_300m.pt \
  --checkpoint weights/19LA_Student.pt --manifest data/test.jsonl \
  --output-dir results/student
```

Teacher inference also requires the paired FlashSR manifest:

```bash
GPU_IDS=0 bash eval_teacher.sh --xlsr-checkpoint weights/xlsr2_300m.pt \
  --checkpoint weights/19LA_Teacher.pt --manifest data/test.jsonl \
  --sr-manifest data/pairs/test.jsonl --output-dir results/teacher
```

Evaluation uses **only the first 64,600 samples (4.0375 s)** after conversion to mono 16-kHz audio; shorter utterances are zero-padded. 

## Acknowledgments

We use [XLS-R/fairseq](https://github.com/facebookresearch/fairseq), [FlashSR](https://github.com/jakeoneijk/FlashSR_Inference), and [RawBoost](https://github.com/TakHemlata/RawBoost-antispoofing). The RawBoost license is retained in `rawboost.py`. Please follow the respective upstream licenses and checkpoint terms.
