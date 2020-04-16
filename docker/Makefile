docker-build:
	cd rascil-no-data;docker build -t timcornwell/rascil-no-data .
	cd rascil-full;docker build -t timcornwell/rascil-full .
	cd rascil-notebook;docker build -t timcornwell/rascil-notebook .

docker-build-no-cache:
	cd rascil-no-data;docker build --no-cache -t timcornwell/rascil-no-data .
	cd rascil-full;docker build --no-cache -t timcornwell/rascil-full .
	cd rascil-notebook;docker build --no-cache -t timcornwell/rascil-notebook .

docker-build-stable:
	cd rascil-no-data;docker build --no-cache -t timcornwell/rascil-no-data:stable .
	cd rascil-full;docker build --no-cache -t timcornwell/rascil-full:stable .
	cd rascil-notebook;docker build --no-cache -t timcornwell/rascil-notebook:stable .

docker-push-stable:
	docker push timcornwell/rascil-no-data:stable
	docker push timcornwell/rascil-full:stable
	docker push timcornwell/rascil-notebook:stable
