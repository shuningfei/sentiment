# !/bin/bash

rm results.txt
touch results.txt

cd cnn

python3 build_vocab_embed.py

echo "build data..."

python3 build_data.py

mkdir feature_maps_test_static
mkdir feature_maps_test_tuned

echo "sentiment classification for [-phrases]..."
echo "sentiment classification for [-phrases]" >> ../results.txt
echo "cnn-static..."
echo "cnn-static" >> ../results.txt
python3 execute_fm.py test static >> ../results.txt

echo "cnn-tuned..."
echo "cnn-tuned" >> ../results.txt
python3 execute_fm.py test tuned  >> ../results.txt

cd ../evaluation
echo "Exaction of subjective phrases for [+phrases,+unseen]..."
echo "Exaction of subjective phrases for [+phrases,+unseen]" >> ../results.txt
echo "cnn-static..."
echo "cnn-static" >> ../results.txt
python ngramRank.py static >> ../results.txt
echo "cnn-tuned..."
echo "cnn-tuned" >> ../results.txt
python ngramRank.py tuned >> ../results.txt