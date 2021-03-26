## TinyMS Dockerfile Repository

This folder hosts all the `Dockerfile` to build TinyMS container images with various versions and corresponding hardware backends.

> **NOTICE:** Currently TinyMS only supports `CPU` hardware backend.

### TinyMS docker build command

| Hardware Platform | Docker Image Tag | Build Command |
| :---------------- | :------ | :------------ |
| CPU | `x.y.z` | cd tinyms/x.y.z && docker build . -t tinyms/tinyms:x.y.z |
|  | `x.y.z-jupyter` | cd tinyms/x.y.z-jupyter && docker build . -t tinyms/tinyms:x.y.z-jupyter |

> **NOTICE:** The `x.y.z` version shown above should be replaced with the real version number.
