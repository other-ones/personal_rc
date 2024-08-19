export current_path=$PWD;
cd ../../;
git add .;
git commit -m "update";
git push origin main;
cd $current_path;
