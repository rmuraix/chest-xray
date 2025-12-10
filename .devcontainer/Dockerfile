FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV UV_LINK_MODE=copy
# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list