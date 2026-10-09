# Розгортання PARCS на GCP (коротко)

Проєкт `parks-510305`, зона `europe-central2-a`, 5 × n1-standard-1 (Debian 12), Docker через startup-скрипт.

```bash
# правило фаєрвола (тільки для лабораторної! після роботи видалити)
gcloud compute firewall-rules create allow-all --allow=tcp,udp,icmp --source-ranges=0.0.0.0/0

# master (веб-інтерфейс на порту 8080) і воркери створюються з Debian-образу;
# startup-скрипт ставить docker.io та запускає контейнер hummer12007/parcs-node
# з --network=host і PARCS_ARGS (адреса master для воркерів).
gcloud compute instances create master worker1 worker2 worker3 worker4 \
    --zone=europe-central2-a --machine-type=n1-standard-1 \
    --image-family=debian-12 --image-project=debian-cloud \
    --metadata-from-file=startup-script=startup.sh   # master/worker -- різні startup.sh

# після завершення експериментів
gcloud compute instances delete master worker1 worker2 worker3 worker4 --zone=europe-central2-a
gcloud compute firewall-rules delete allow-all
```

Запуск завдання: веб-інтерфейс master (`http://<IP>:8080`) -> Jobs -> Add -> завантажити
`solution/kmeans_solver.py` (як solution.py), вхідний файл, кількість воркерів.
