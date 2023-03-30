#!/bin/bash
sudo mv helloworld.service /etc/systemd/system/helloworld.service


echo Restart Daemon
sudo systemctl daemon-reload
sudo systemctl start helloworld
sudo systemctl enable helloworld

echo ------------
echo test connection
curl localhost:8000
echo ------------

