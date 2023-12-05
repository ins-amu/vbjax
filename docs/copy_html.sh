
rm -rf _modules
rm -rf _sources
rm -rf _static
make html 
cp -r build/html/* .
rm -rf build
