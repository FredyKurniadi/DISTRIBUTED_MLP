# DISTRIBUTED_MLP

Simulasi distributed MLP di 1 mesin dengan multi-proses, komunikasi TCP JSON, dan pembagian komputasi per-layer menjadi blok matrix multiply.

## Versi Python
- Digunakan: Python 3.9.13 (local venv di folder DISTRIBUTED_MLP)

## Fitur Inti
- Teacher-student synthetic task dengan aktivasi ReLU.
- Coordinator membagi task matmul ke worker via TCP.
- Worker menghitung blok lalu mengirim result ke coordinator.
- Setiap worker menyimpan state lokal ke file JSON terpisah.

## Struktur Penting
- `configs/`: konfigurasi model dan distributed runtime.
- `src/distributed/`: protocol, worker, coordinator, partitioner, assembler.
- `src/mlp/`: model dan training distributed.
- `runtime/worker_state/`: state JSON tiap worker.
- `scripts/`: setup dan run command.

Lihat `QUICKSTART.md` untuk langkah menjalankan.

## Jalankan Semua Sekaligus
Setelah setup environment, jalankan:

```powershell
.\scripts\run_all.ps1
```
