rm -r files/
rm -r images/
python3 main.py -m 3DQN
python3 main.py --test -m actuated
python3 main.py --test -m static_secured --load-scenario
python3 main.py --test -m unsecured --load-scenario
python3 main.py --test -m 3DQN --load-scenario
python graphs.py --test
python3 main.py --full-test -m actuated
python3 main.py --full-test -m 3DQN --load-scenario
python graphs.py --full-test