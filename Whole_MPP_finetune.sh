#!/bin/bash
# MPP

#!/bin/bash
for dataset in BACE BBBP ClinTox SIDER AMES hERG Tox21 HIV ; do
  echo "Starting ${dataset}..."
   
  BS_OPT=""
  case "${dataset}" in
    Tox21|hERG|AMES|MUV|HIV) BS_OPT="--batch_size 64" ;;
  esac

  python MPP_finetune.py --dataset "${dataset}" --device 0 --dropout 0.3 ${BS_OPT} \
done
