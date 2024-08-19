export current_path=$PWD;
echo $current_path;
cd ../../;
git pull origin main;
cd $current_path;