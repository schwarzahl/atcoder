for file in `ls testcases`
do
  cat testcases/${file} | java -jar cp_ide.jar > output/${file}
  cat testcases/${file} output/${file} | python evaluator.py
done
