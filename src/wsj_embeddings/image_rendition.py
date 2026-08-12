"""Deterministic, content-free preparation of hosted header-image inputs."""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import stat
import warnings
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

MAX_HOSTED_IMAGE_BYTES = 5_000_000
MAX_HOSTED_IMAGE_PIXELS = 20_000_000
MAX_SAFE_DECODE_PIXELS = 40_000_000
SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
SOURCE_BYTES_TRANSFORM_ID = "exact-source-bytes-v1"
PRODUCTION_IMAGE_INPUT_RULES = (
    "static-jpeg-png-webp-max5000000b-max20000000px-decode-max40000000px-"
    "exact-source-v1"
)
PRODUCTION_IMAGE_TRANSFORM_ID = (
    "pillow-11.3.0-exif-transpose-alpha-white-rgb-jpeg-q85-420-lanczos-"
    "if-over20000000px-optimize0-progressive0-metadata-none-v1"
)
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
_STAGE_FLAGS = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
)


class ImageCodecError(RuntimeError):
    """A stable local image decoding or encoding disposition."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ImageInfo:
    """Content-free facts from a complete, safe image decode."""

    format: str
    width: int
    height: int
    frames: int = 1

    @property
    def pixels(self) -> int:
        return self.width * self.height


class ImageCodec(Protocol):
    """Decode and render behind an injectable deterministic implementation."""

    input_rules: str
    transform_id: str

    def inspect(self, data: bytes, *, max_decode_pixels: int) -> ImageInfo: ...

    def render_jpeg(
        self,
        data: bytes,
        *,
        width: int,
        height: int,
    ) -> bytes: ...


class PillowImageCodec:
    """Pinned production decoder and deterministic JPEG rendition encoder."""

    input_rules = PRODUCTION_IMAGE_INPUT_RULES
    transform_id = PRODUCTION_IMAGE_TRANSFORM_ID

    def __init__(self) -> None:
        try:
            import PIL
            from PIL import Image, ImageOps, UnidentifiedImageError, features
        except ImportError as error:
            raise ImageCodecError("image_codec_unavailable") from error
        if PIL.__version__ != "11.3.0":
            raise ImageCodecError("image_codec_unavailable")
        self._image = Image
        self._image_ops = ImageOps
        self._unidentified_error = UnidentifiedImageError
        self._bomb_error = Image.DecompressionBombError
        self._bomb_warning = Image.DecompressionBombWarning
        self.transform_id = _pillow_transform_id(features)

    def inspect(self, data: bytes, *, max_decode_pixels: int) -> ImageInfo:
        """Fully verify and decode one bounded in-memory image."""

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", self._bomb_warning)
                with self._image.open(io.BytesIO(data)) as initial:
                    image_format = str(initial.format or "").upper()
                    frames = int(getattr(initial, "n_frames", 1))
                    width, height = initial.size
                    if width * height > max_decode_pixels:
                        raise ImageCodecError("unsafe_image")
                    initial.verify()
                with self._image.open(io.BytesIO(data)) as decoded:
                    oriented = self._image_ops.exif_transpose(decoded)
                    oriented.load()
                    oriented_width, oriented_height = oriented.size
                    if oriented_width * oriented_height > max_decode_pixels:
                        raise ImageCodecError("unsafe_image")
        except ImageCodecError:
            raise
        except (self._bomb_error, self._bomb_warning) as error:
            raise ImageCodecError("unsafe_image") from error
        except (OSError, SyntaxError, ValueError, self._unidentified_error) as error:
            raise ImageCodecError("corrupt_image") from error
        return ImageInfo(image_format, oriented_width, oriented_height, frames)

    def render_jpeg(
        self,
        data: bytes,
        *,
        width: int,
        height: int,
    ) -> bytes:
        """Render metadata-free RGB JPEG with one pinned encoder contract."""

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", self._bomb_warning)
                with self._image.open(io.BytesIO(data)) as decoded:
                    image = self._image_ops.exif_transpose(decoded)
                    image.load()
                    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                        rgba = image.convert("RGBA")
                        background = self._image.new("RGBA", rgba.size, "white")
                        image = self._image.alpha_composite(background, rgba).convert(
                            "RGB"
                        )
                    else:
                        image = image.convert("RGB")
                    if image.size != (width, height):
                        image = image.resize(
                            (width, height),
                            resample=self._image.Resampling.LANCZOS,
                        )
                    output = io.BytesIO()
                    image.save(
                        output,
                        format="JPEG",
                        quality=85,
                        subsampling=2,
                        optimize=False,
                        progressive=False,
                    )
        except (self._bomb_error, self._bomb_warning) as error:
            raise ImageCodecError("unsafe_image") from error
        except (OSError, SyntaxError, ValueError, self._unidentified_error) as error:
            raise ImageCodecError("image_encode_failure") from error
        return output.getvalue()


def _pillow_transform_id(features: object) -> str:
    """Bind deterministic transform meaning to the linked Pillow codec build."""

    try:
        if not features.check_codec("jpg") or not features.check_codec("zlib"):
            raise ImageCodecError("image_codec_unavailable")
        if not features.check("webp"):
            raise ImageCodecError("image_codec_unavailable")
        jpeg = _safe_build_version(features.version_codec("jpg"))
        zlib = _safe_build_version(features.version_codec("zlib"))
        webp = _safe_build_version(features.version_module("webp"))
        turbo = (
            _safe_build_version(features.version_feature("libjpeg_turbo"))
            if features.check_feature("libjpeg_turbo")
            else "none"
        )
    except ImageCodecError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ImageCodecError("image_codec_unavailable") from error
    return (
        f"{PRODUCTION_IMAGE_TRANSFORM_ID}-build-jpeg-{jpeg}-turbo-{turbo}-"
        f"zlib-{zlib}-webp-{webp}"
    )


def _safe_build_version(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._+\-]+", value):
        raise ImageCodecError("image_codec_unavailable")
    return value


class FixturePassthroughImageCodec:
    """Offline generated-fixture codec; never used by hosted production runs."""

    input_rules = PRODUCTION_IMAGE_INPUT_RULES
    transform_id = PRODUCTION_IMAGE_TRANSFORM_ID

    def inspect(self, data: bytes, *, max_decode_pixels: int) -> ImageInfo:
        del max_decode_pixels
        if not data:
            raise ImageCodecError("corrupt_image")
        return ImageInfo("PNG", 1, 1)

    def render_jpeg(
        self,
        data: bytes,
        *,
        width: int,
        height: int,
    ) -> bytes:
        del data, width, height
        raise ImageCodecError("image_encode_failure")


@dataclass(frozen=True, slots=True)
class PreparedImageInput:
    """Exact bytes and content-free provenance for one hosted image request."""

    data: bytes
    source_sha256: str
    source_info: ImageInfo
    embedded_input_sha256: str
    embedded_info: ImageInfo
    transform_id: str
    rendition_relative_path: str | None = None
    rendition_identity: RenditionIdentity | None = None


@dataclass(frozen=True, slots=True)
class RenditionIdentity:
    """Held namespace identities for revalidation at publication boundary."""

    relative_path: str
    rendition_directory: tuple[int, int]
    transform_directory: tuple[int, int]


def prepare_image_input(
    source_data: bytes,
    *,
    codec: ImageCodec,
    expected_input_rules: str,
    expected_transform_id: str,
    install_rendition: Callable[[bytes, ImageInfo, str], RenditionIdentity]
    | None = None,
) -> PreparedImageInput:
    """Choose exact source bytes or one verified deterministic rendition."""

    if (
        codec.input_rules != expected_input_rules
        or codec.transform_id != expected_transform_id
    ):
        raise ImageCodecError("ambiguous_image_configuration")
    source_sha256 = hashlib.sha256(source_data).hexdigest()
    source_info = _validated_info(
        codec.inspect(source_data, max_decode_pixels=MAX_SAFE_DECODE_PIXELS)
    )
    if (
        len(source_data) <= MAX_HOSTED_IMAGE_BYTES
        and source_info.pixels <= MAX_HOSTED_IMAGE_PIXELS
    ):
        return PreparedImageInput(
            data=source_data,
            source_sha256=source_sha256,
            source_info=source_info,
            embedded_input_sha256=source_sha256,
            embedded_info=source_info,
            transform_id=SOURCE_BYTES_TRANSFORM_ID,
        )
    width, height = scaled_image_dimensions(source_info.width, source_info.height)
    derived = codec.render_jpeg(source_data, width=width, height=height)
    derived_info = _validated_info(
        codec.inspect(derived, max_decode_pixels=MAX_SAFE_DECODE_PIXELS)
    )
    if (
        derived_info.format != "JPEG"
        or len(derived) > MAX_HOSTED_IMAGE_BYTES
        or derived_info.pixels > MAX_HOSTED_IMAGE_PIXELS
    ):
        raise ImageCodecError("derived_image_oversized")
    derived_sha256 = hashlib.sha256(derived).hexdigest()
    if install_rendition is None:
        raise ImageCodecError("rendition_installation_unavailable")
    installation = install_rendition(derived, derived_info, derived_sha256)
    return PreparedImageInput(
        data=derived,
        source_sha256=source_sha256,
        source_info=source_info,
        embedded_input_sha256=derived_sha256,
        embedded_info=derived_info,
        transform_id=codec.transform_id,
        rendition_relative_path=installation.relative_path,
        rendition_identity=installation,
    )


def _validated_info(info: ImageInfo) -> ImageInfo:
    image_format = str(info.format).upper()
    if image_format not in SUPPORTED_IMAGE_FORMATS:
        raise ImageCodecError("unsupported_image")
    if info.frames != 1:
        raise ImageCodecError("unsupported_image")
    if info.width < 1 or info.height < 1:
        raise ImageCodecError("corrupt_image")
    if info.pixels > MAX_SAFE_DECODE_PIXELS:
        raise ImageCodecError("unsafe_image")
    return ImageInfo(image_format, info.width, info.height, info.frames)


def scaled_image_dimensions(width: int, height: int) -> tuple[int, int]:
    """Return the exact deterministic aspect scale for the hosted pixel ceiling."""

    if width < 1 or height < 1:
        raise ImageCodecError("corrupt_image")
    if width * height <= MAX_HOSTED_IMAGE_PIXELS:
        return width, height
    scale = math.sqrt(MAX_HOSTED_IMAGE_PIXELS / (width * height))
    width = max(1, int(width * scale))
    height = max(1, int(height * scale))
    while width * height > MAX_HOSTED_IMAGE_PIXELS:
        if width >= height:
            width -= 1
        else:
            height -= 1
    return width, height


def install_image_rendition(
    output_descriptor: int,
    data: bytes,
    info: ImageInfo,
    derived_sha256: str,
    *,
    transform_id: str,
    codec: ImageCodec,
) -> RenditionIdentity:
    """Stage, reopen, verify, and atomically install one immutable rendition."""

    transform_namespace = hashlib.sha256(transform_id.encode()).hexdigest()
    rendition_descriptor = _open_or_create_directory(
        output_descriptor,
        "renditions",
    )
    transform_descriptor: int | None = None
    temporary_name = f".{derived_sha256}.{uuid4().hex}.tmp"
    final_name = f"{derived_sha256}.jpg"
    try:
        transform_descriptor = _open_or_create_directory(
            rendition_descriptor,
            transform_namespace,
        )
        try:
            stage_descriptor = os.open(
                temporary_name,
                _STAGE_FLAGS,
                0o600,
                dir_fd=transform_descriptor,
            )
        except OSError as error:
            raise ImageCodecError("unsafe_rendition_output") from error
        try:
            stage_stat = os.fstat(stage_descriptor)
            if not stat.S_ISREG(stage_stat.st_mode) or stage_stat.st_nlink != 1:
                raise ImageCodecError("unsafe_rendition_output")
            view = memoryview(data)
            while view:
                written = os.write(stage_descriptor, view)
                if written < 1:
                    raise ImageCodecError("unsafe_rendition_output")
                view = view[written:]
            os.fsync(stage_descriptor)
        except OSError as error:
            raise ImageCodecError("unsafe_rendition_output") from error
        finally:
            os.close(stage_descriptor)
        _verify_rendition_leaf(
            transform_descriptor,
            temporary_name,
            data,
            info,
            codec,
        )
        try:
            os.link(
                temporary_name,
                final_name,
                src_dir_fd=transform_descriptor,
                dst_dir_fd=transform_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            _verify_rendition_leaf(
                transform_descriptor,
                final_name,
                data,
                info,
                codec,
            )
        except OSError as error:
            raise ImageCodecError("unsafe_rendition_output") from error
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=transform_descriptor)
        try:
            os.fsync(transform_descriptor)
        except OSError as error:
            raise ImageCodecError("unsafe_rendition_output") from error
        _verify_rendition_leaf(
            transform_descriptor,
            final_name,
            data,
            info,
            codec,
        )
        installation = RenditionIdentity(
            relative_path=f"renditions/{transform_namespace}/{final_name}",
            rendition_directory=_directory_identity(rendition_descriptor),
            transform_directory=_directory_identity(transform_descriptor),
        )
        _verify_rendition_chain(
            output_descriptor,
            installation,
            final_name,
            data,
            info,
            codec,
        )
    finally:
        if transform_descriptor is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=transform_descriptor)
            os.close(transform_descriptor)
        os.close(rendition_descriptor)
    return installation


def verify_image_rendition(
    output_descriptor: int,
    installation: RenditionIdentity,
    data: bytes,
    info: ImageInfo,
    codec: ImageCodec,
) -> None:
    """Revalidate an installed rendition immediately before catalog mutation."""

    final_name = installation.relative_path.rsplit("/", 1)[-1]
    _verify_rendition_chain(
        output_descriptor,
        installation,
        final_name,
        data,
        info,
        codec,
    )


def cleanup_orphan_image_renditions(
    output_descriptor: int,
    is_referenced: Callable[[str], bool],
    *,
    after_unlink: Callable[[str], None] | None = None,
) -> int:
    """Delete only safe unreferenced final renditions from anchored namespaces."""

    try:
        rendition_descriptor = os.open(
            "renditions", _DIRECTORY_FLAGS, dir_fd=output_descriptor
        )
    except FileNotFoundError:
        return 0
    except OSError as error:
        raise ImageCodecError("unsafe_rendition_output") from error
    removed = 0
    try:
        with os.scandir(rendition_descriptor) as namespaces:
            for namespace in namespaces:
                if (
                    namespace.is_symlink()
                    or not namespace.is_dir(follow_symlinks=False)
                    or re.fullmatch(r"[0-9a-f]{64}", namespace.name) is None
                ):
                    continue
                try:
                    namespace_descriptor = os.open(
                        namespace.name,
                        _DIRECTORY_FLAGS,
                        dir_fd=rendition_descriptor,
                    )
                except OSError:
                    continue
                try:
                    with os.scandir(namespace_descriptor) as entries:
                        for entry in entries:
                            if (
                                entry.is_symlink()
                                or not entry.is_file(follow_symlinks=False)
                                or re.fullmatch(r"[0-9a-f]{64}\.jpg", entry.name)
                                is None
                            ):
                                continue
                            relative_path = (
                                f"renditions/{namespace.name}/{entry.name}"
                            )
                            if is_referenced(relative_path):
                                continue
                            before = entry.stat(follow_symlinks=False)
                            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                                continue
                            observed = os.stat(
                                entry.name,
                                dir_fd=namespace_descriptor,
                                follow_symlinks=False,
                            )
                            if (before.st_dev, before.st_ino) != (
                                observed.st_dev,
                                observed.st_ino,
                            ):
                                continue
                            os.unlink(entry.name, dir_fd=namespace_descriptor)
                            os.fsync(namespace_descriptor)
                            removed += 1
                            if after_unlink is not None:
                                after_unlink(relative_path)
                finally:
                    os.close(namespace_descriptor)
    except OSError as error:
        raise ImageCodecError("unsafe_rendition_output") from error
    finally:
        os.close(rendition_descriptor)
    return removed


def _open_or_create_directory(parent_descriptor: int, name: str) -> int:
    if not name or name in {".", ".."} or os.path.basename(name) != name:
        raise ImageCodecError("unsafe_rendition_output")
    descriptor: int | None = None
    try:
        with suppress(FileExistsError):
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor)
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        os.fsync(parent_descriptor)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ImageCodecError("unsafe_rendition_output") from error
    if (
        not stat.S_ISDIR(path_stat.st_mode)
        or not stat.S_ISDIR(descriptor_stat.st_mode)
        or (path_stat.st_dev, path_stat.st_ino)
        != (descriptor_stat.st_dev, descriptor_stat.st_ino)
    ):
        os.close(descriptor)
        raise ImageCodecError("unsafe_rendition_output")
    return descriptor


def _verify_rendition_chain(
    output_descriptor: int,
    installation: RenditionIdentity,
    final_name: str,
    expected_data: bytes,
    expected_info: ImageInfo,
    codec: ImageCodec,
) -> None:
    """Re-anchor both namespace directories and final leaf from output root."""

    rendition_descriptor: int | None = None
    transform_descriptor: int | None = None
    try:
        rendition_descriptor = os.open(
            "renditions",
            _DIRECTORY_FLAGS,
            dir_fd=output_descriptor,
        )
        _require_directory_identity(
            rendition_descriptor, installation.rendition_directory
        )
        transform_namespace = installation.relative_path.split("/", 2)[1]
        transform_descriptor = os.open(
            transform_namespace,
            _DIRECTORY_FLAGS,
            dir_fd=rendition_descriptor,
        )
        _require_directory_identity(
            transform_descriptor, installation.transform_directory
        )
        _verify_rendition_leaf(
            transform_descriptor,
            final_name,
            expected_data,
            expected_info,
            codec,
        )
        _reanchor_namespace_identities(output_descriptor, installation)
    except OSError as error:
        raise ImageCodecError("unsafe_rendition_output") from error
    finally:
        if transform_descriptor is not None:
            os.close(transform_descriptor)
        if rendition_descriptor is not None:
            os.close(rendition_descriptor)


def _directory_identity(descriptor: int) -> tuple[int, int]:
    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode):
        raise ImageCodecError("unsafe_rendition_output")
    return observed.st_dev, observed.st_ino


def _require_directory_identity(
    descriptor: int,
    expected: tuple[int, int],
) -> None:
    observed = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or (observed.st_dev, observed.st_ino) != expected
    ):
        raise ImageCodecError("unsafe_rendition_output")


def _reanchor_namespace_identities(
    output_descriptor: int,
    installation: RenditionIdentity,
) -> None:
    rendition_descriptor: int | None = None
    transform_descriptor: int | None = None
    try:
        rendition_descriptor = os.open(
            "renditions", _DIRECTORY_FLAGS, dir_fd=output_descriptor
        )
        _require_directory_identity(
            rendition_descriptor, installation.rendition_directory
        )
        transform_descriptor = os.open(
            installation.relative_path.split("/", 2)[1],
            _DIRECTORY_FLAGS,
            dir_fd=rendition_descriptor,
        )
        _require_directory_identity(
            transform_descriptor, installation.transform_directory
        )
    except OSError as error:
        raise ImageCodecError("unsafe_rendition_output") from error
    finally:
        if transform_descriptor is not None:
            os.close(transform_descriptor)
        if rendition_descriptor is not None:
            os.close(rendition_descriptor)


def _verify_rendition_leaf(
    directory_descriptor: int,
    name: str,
    expected_data: bytes,
    expected_info: ImageInfo,
    codec: ImageCodec,
) -> None:
    try:
        descriptor = os.open(name, _READ_FLAGS, dir_fd=directory_descriptor)
    except OSError as error:
        raise ImageCodecError("unsafe_rendition_output") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != len(expected_data)
        ):
            raise ImageCodecError("unsafe_rendition_output")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            observed = stream.read(MAX_HOSTED_IMAGE_BYTES + 1)
        observed_info = _validated_info(
            codec.inspect(observed, max_decode_pixels=MAX_HOSTED_IMAGE_PIXELS)
        )
        after = os.fstat(descriptor)
        path_stat = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ImageCodecError("unsafe_rendition_output") from error
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or after.st_size != len(expected_data)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or not stat.S_ISREG(path_stat.st_mode)
        or path_stat.st_nlink != 1
        or (path_stat.st_dev, path_stat.st_ino) != (after.st_dev, after.st_ino)
        or observed != expected_data
        or hashlib.sha256(observed).hexdigest()
        != hashlib.sha256(expected_data).hexdigest()
        or observed_info != expected_info
    ):
        raise ImageCodecError("unsafe_rendition_output")
