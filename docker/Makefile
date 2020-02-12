docker:
	cd rascil-no-data;docker build -t timcornwell/rascil-no-data .
	cd rascil-full;docker build -t timcornwell/rascil-full .

docker-no-cache:
	cd rascil-no-data;docker build --no-cache -t timcornwell/rascil-no-data .
	cd rascil-full;docker build --no-cache -t timcornwell/rascil-full .

docker-push:
	docker push timcornwell/rascil-no-data
	docker push timcornwell/rascil-full