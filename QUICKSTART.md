# Quickstart

## 1. Setup environment lokal proyek
```powershell
.\scripts\setup_all.ps1
```

## 2. Jalankan worker (masing-masing di terminal berbeda)
Terminal A:
```powershell
.\scripts\run_worker.ps1 -WorkerId 1 -Port 6101
```

Terminal B:
```powershell
.\scripts\run_worker.ps1 -WorkerId 2 -Port 6102
```

Terminal C:
```powershell
.\scripts\run_worker.ps1 -WorkerId 3 -Port 6103
```

## 3. Jalankan training (coordinator ada di proses ini)
Terminal D:
```powershell
.\scripts\run_train.ps1
```

## 4. Cek state worker
Lihat file JSON di:
- `runtime/worker_state/worker_001.json`
- `runtime/worker_state/worker_002.json`
- `runtime/worker_state/worker_003.json`

## Opsi cepat: jalankan semuanya sekaligus
```powershell
.\scripts\run_all.ps1
```
