his_data=historical_data.csv
train_data=train_data.csv
dev_data=validation_data.csv
num_sam=`wc $his_data|awk '{print $1-1}'`
num_train=`wc $his_data|awk '{printf("%d", $1*0.9)}'`
num_dev=$(( num_sam - num_train ))
#echo $num_sam $num_train $num_dev
grep -v market_id $his_data |perl -MList::Util=shuffle -wne 'print shuffle <>;' > all.csv
head -n 1 $his_data > $train_data
head -n 1 $his_data > $dev_data
head -n $num_train all.csv >> $train_data
tail -n $num_dev all.csv >> $dev_data

