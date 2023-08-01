#!/bin/sh

for PREFIX in episode{1,2,3,4,5,6} backstroke; do
    iconv -f cp1252 -t utf8 ${PREFIX}.en.srt | python3 srt2tsv.py > ${PREFIX}.en.tsv
    iconv -f gbk -t utf8 ${PREFIX}.zh.srt | python3 srt2tsv.py > ${PREFIX}.zh.tsv
done

python align.py episode1.{zh,en}.tsv 120.453 114.52 7776.278 7457.071 > episode1.zh-en.tsv
python align.py episode2.{zh,en}.tsv 130.082 130.08 7707.793 7707.396 > episode2.zh-en.tsv
python align.py episode3.{zh,en}.tsv 177.046 179.88 7515.126 7514.479 > episode3.zh-en.tsv
cut -f1-3 episode3.zh-en.tsv > /tmp/episode3.zh.tsv
python3 align.py -f /tmp/episode3.zh.tsv backstroke.en.tsv 177.046 186.852 7515.126 7839.166 > backstroke.zh-en.tsv
python align.py episode4.{zh,en}.tsv 158.9 160.68 6811.652 6813.797 > episode4.zh-en.tsv
python align.py episode5.{zh,en}.tsv 203.9 204.88 6938.657 6939.276 > episode5.zh-en.tsv
python align.py episode6.{zh,en}.tsv 141.75 148.849 7246.475 7557.109 > episode6.zh-en.tsv

cat episode[12456].zh-en.tsv | cut -f3 | perl tokenize.pl > small.zh
cat episode[12456].zh-en.tsv | cut -f4 | perl tokenize.pl > small.en
sed -ne '1,399p' episode3.zh-en.tsv | cut -f3 | perl tokenize.pl > dev.zh
sed -ne '400,$p' episode3.zh-en.tsv | cut -f3 | perl tokenize.pl > test.zh
sed -ne '1,399p' episode3.zh-en.tsv | cut -f4 | perl tokenize.pl > dev.reference.en
sed -ne '400,$p' episode3.zh-en.tsv | cut -f4 | perl tokenize.pl > test.reference.en
sed -ne '1,399p' backstroke.zh-en.tsv | cut -f4 | perl tokenize.pl > dev.backstroke.en
sed -ne '400,$p' backstroke.zh-en.tsv | cut -f4 | perl tokenize.pl > test.backstroke.en

curl -LO 'https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-zh_cn.txt.zip'
unzip -l en-zh_cn.txt.zip
paste OpenSubtitles.en-zh_cn.{zh_cn,en,ids} | egrep '(zh_cn|en)/2004/' | cut -f1-2 | perl tokenize.pl > other.zh-en
(for I in `seq 1 10`; do cat small.zh; done; cut -f1 large.zh-en) > large.zh
(for I in `seq 1 10`; do cat small.en; done; cut -f2 large.zh-en) > large.en

mv {small,large}.{zh,en} {dev,test}.{zh,reference.en,backstroke.en} ../data
