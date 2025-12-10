# Federated MTL-LoRA with Flower: Integration & Testing

This guide explains how to pull the federated training code into your own branch and how to run a quick smoke test.

## 1) Den Code in deinen Branch übernehmen

Die Federated-MLoRA-Dateien liegen bereits im Repository (`federated_mlora.py`, `src/federated/*`). Du hast mehrere Optionen, sie in deinen Arbeitsbranch zu holen:

- **Aktuelle Branches mergen**: `git checkout <dein-branch>` und `git merge work` (oder den Branch-Namen, auf dem diese Änderungen liegen).
- **Commit cherry-picken**: `git cherry-pick <commit-hash>` falls du nur den letzten Commit übernehmen willst.
- **Copy/Paste** ist möglich, aber vermeiden Git-Konflikte und setze die Datei-Historie fort, wenn du über `merge` oder `cherry-pick` gehst.

## 2) Umgebung einrichten

```bash
conda create -n mtl-lora python==3.10 -y
conda activate mtl-lora
pip install -r requirements.txt
```

Stelle sicher, dass du ein HF-Basismodell (z. B. Llama-2/3) herunterladen kannst und dass deine Dataset-Pfade lokal oder von Hugging Face erreichbar sind.

## 3) Flower-Server starten

```bash
python federated_mlora.py server --server_address 0.0.0.0:8080 --num_rounds 3
```

## 4) Flower-Clients starten

Starte in separaten Terminals (oder getrennten Maschinen) pro Daten-Split einen Client:

```bash
python federated_mlora.py client \
  --server_address 0.0.0.0:8080 \
  --base_model meta-llama/Meta-Llama-3-8B \
  --data_path /pfad/zum/datensatz.json \
  --adapter_name mlora \
  --lora_target_modules "('q_proj','k_proj','v_proj','o_proj')" \
  --batch_size 2 --num_epochs 1 --learning_rate 5e-5 \
  --cutoff_len 256 --val_split 0.1 --train_on_inputs
```

Wichtig:
- Nutze pro Client unterschiedliche Daten-Splits oder Shards.
- Wenn dein Dataset `task_id`-Felder besitzt, werden diese automatisch als `lambda_index` ins Modell gespeist (benötigt für MTL-/MoE-LoRA).

## 5) Schneller Funktionstest (ohne großes Training)

Führe einen Syntax-/Import-Check aus, der keine GPUs oder Modelle herunterlädt:

```bash
python -m compileall federated_mlora.py src/federated
```

So stellst du sicher, dass die Flower-Integration und die Hilfsfunktionen korrekt installiert sind.

## 6) Training überwachen

- Die Flower-Strategie ist standardmäßig `FedAvg`. Du kannst Lernrate und lokale Epochen pro Runde über die Server-Konfiguration (`fit_config_fn`) oder Client-Argumente steuern.
- Im Client-Log erscheinen Verlustwerte pro Runde. Nutze zusätzlich Metriken aus `evaluate`, falls du Validierungsdaten bereitstellst.

## 7) Häufige Stolpersteine

- **Tokenizer/Pad-Token**: Der Client setzt `tokenizer.pad_token_id = tokenizer.unk_token_id` und `padding_side='right'`. Halte das zwischen Clients konsistent.
- **Daten-Caching**: Falls `--cache_dir` gesetzt ist, werden tokenisierte Splits lokal gespeichert, um Startzeiten zu verkürzen.
- **GPU vs. CPU**: Wenn keine GPU verfügbar ist, fällt der Client automatisch auf CPU zurück; Training kann dann deutlich langsamer sein.

Viel Erfolg beim föderierten Fine-Tuning! Bei Fragen: einfach den oben beschriebenen Merge-/Cherry-Pick-Schritt wiederholen und Logs teilen.
