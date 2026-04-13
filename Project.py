!pip install qiskit qiskit-aer opencv-python-headless scikit-image scikit-learn numpy matplotlib pycryptodome pandas reedsolo
import os
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error, mean_absolute_error

from reedsolo import RSCodec

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def load_images(out_dir='images'):
    os.makedirs(out_dir, exist_ok=True)

    input_paths = ['/content/photo1.jpg', '/content/photo2.jpg', '/content/photo3.jpg', '/content/photo4.jpg']
    output_paths = []

    for i, input_path in enumerate(input_paths):
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found. Creating a dummy image.")
            dummy_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            Image.fromarray(dummy_img).save(input_path)

    for input_path in input_paths:
        if not os.path.exists(input_path):
            print(f"Warning: Image file not found at {input_path}. Skipping.")
            continue

        try:
            img = np.array(Image.open(input_path).convert('RGB'), dtype=np.uint8)

            if img.shape[0] != 512 or img.shape[1] != 512:
                print(f"Resizing {input_path} from {img.shape} to (512, 512)...")
                img = np.array(Image.fromarray(img).resize((512, 512), Image.Resampling.LANCZOS), dtype=np.uint8)

            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(out_dir, f'{base_name}.png')

            print(f"Loaded {input_path} min: {img.min()}, max: {img.max()}, dtype: {img.dtype}, shape: {img.shape}")
            Image.fromarray(img).save(output_path, 'PNG')
            print(f"Saved as {output_path}")
            output_paths.append(output_path)

        except Exception as e:
            print(f"Error loading {input_path}: {e}")

    if not output_paths:
        raise FileNotFoundError("No valid input images found or created. Please upload 'photo1.jpg', 'photo2.jpg', 'photo3.jpg', and 'photo4.jpg' to /content/")

    return output_paths


def aes_gcm_encrypt(plaintext_bytes: bytes, key: bytes = None):
    if key is None:
        key = get_random_bytes(16)
    iv = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
    ct, tag = cipher.encrypt_and_digest(plaintext_bytes)
    return {'key': key, 'iv': iv, 'ct': ct, 'tag': tag}

def aes_gcm_decrypt(ct: bytes, key: bytes, iv: bytes, tag: bytes):
    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
    return cipher.decrypt_and_verify(ct, tag)


def lsb_steganography(image_path: str, message: str, output_path: str):
    img = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)

    binary_message = ''.join([format(ord(c), '08b') for c in message]) + '1111111111111110'
    data_index = 0
    rows, cols = img.shape[:2]

    if len(binary_message) > rows * cols:
        raise ValueError(f"Message is too large ({len(binary_message)} bits) for image ({rows*cols} pixels)")

    for i in range(rows):
        for j in range(cols):
            if data_index < len(binary_message):
                img[i, j, 0] = np.uint8((img[i, j, 0] & 254) | int(binary_message[data_index]))
                data_index += 1
            else:
                break
        if data_index >= len(binary_message):
            break

    Image.fromarray(img).save(output_path, 'PNG')
    return img

def extract_lsb_steganography(stego_path: str, orig_message_len: int):
    try:
        stego_img = np.array(Image.open(stego_path).convert('RGB'), dtype=np.uint8)
    except Exception as e:
        print(f"PIL failed to open {stego_path} ({e}), trying OpenCV...")
        stego_img_bgr = cv2.imread(stego_path)
        if stego_img_bgr is None:
            raise IOError(f"Failed to open image {stego_path} with both PIL and OpenCV.")
        stego_img = cv2.cvtColor(stego_img_bgr, cv2.COLOR_BGR2RGB)

    binary_message = ''
    rows, cols = stego_img.shape[:2]

    max_bits = (orig_message_len * 8) + 16

    for i in range(rows):
        for j in range(cols):
            binary_message += str(stego_img[i, j, 0] & 1)

            if len(binary_message) >= 16 and binary_message[-16:] == '1111111111111110':
                break

            if len(binary_message) > max_bits + 1024:
                print("Terminator not found, stopping extraction early.")
                break

        if len(binary_message) >= 16 and binary_message[-16:] == '1111111111111110':
            break

    terminator_pos = binary_message.find('1111111111111110')
    if terminator_pos != -1:
        message_bits = binary_message[:terminator_pos]
    else:
        message_bits = binary_message

    message = ''
    for j in range(0, len(message_bits), 8):
        byte_str = message_bits[j:j+8]
        if len(byte_str) == 8:
            try:
                message += chr(int(byte_str, 2))
            except ValueError:
                message += '?'

    return message[:orig_message_len]


def embed_watermark_dct_from_text(image: np.ndarray, watermark_text: str, strength: int = 75):
    watermark_bits = []
    for ch in watermark_text:
        bits = format(ord(ch), '08b')
        watermark_bits.extend([int(b) for b in bits])

    rs = RSCodec(10)
    watermark_bytes = ''.join([chr(int(''.join(map(str, watermark_bits[i:i+8])), 2)) for i in range(0, len(watermark_bits), 8)]).encode('latin-1')
    rs_encoded = rs.encode(watermark_bytes)
    encoded_bits = [int(b) for byte in rs_encoded for b in format(byte, '08b')]

    repetition_factor = 7
    final_bits = []
    for bit in encoded_bits:
        final_bits.extend([bit] * repetition_factor)

    watermarked = embed_watermark_dct(image, final_bits, strength)
    return watermarked, watermark_bits, final_bits

def embed_watermark_dct(image: np.ndarray, watermark_bits: List[int], strength: int = 75):
    watermarked = image.copy().astype(np.float32)
    h, w = watermarked.shape[:2]

    coeff_positions = [(1, 2), (2, 1), (2, 2)]
    channel_to_embed = 2

    bit_idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if bit_idx >= len(watermark_bits):
                break
            block = watermarked[i:i+8, j:j+8, channel_to_embed]
            if block.shape[0] != 8 or block.shape[1] != 8:
                continue

            dct_block = cv2.dct(block)

            for pos in coeff_positions:
                if bit_idx < len(watermark_bits):
                    if watermark_bits[bit_idx] == 1:
                        dct_block[pos] += strength
                    else:
                        dct_block[pos] -= strength
                    bit_idx += 1

            idct_block = cv2.idct(dct_block)
            watermarked[i:i+8, j:j+8, channel_to_embed] = idct_block
        if bit_idx >= len(watermark_bits):
            break

    if bit_idx < len(watermark_bits):
        print(f"Warning: Only embedded {bit_idx} of {len(watermark_bits)} bits.")

    return np.clip(watermarked, 0, 255).astype(np.uint8)

def extract_watermark_dct_to_bits(image: np.ndarray, num_bits: int, repetition_factor: int = 7) -> List[int]:
    h, w = image.shape[:2]
    extracted_bits_repeated = []
    coeff_positions = [(1, 2), (2, 1), (2, 2)]
    channel_to_extract = 2

    coeff_magnitudes = []
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = image[i:i+8, j:j+8, channel_to_extract].astype(np.float32)
            if block.shape[0] != 8 or block.shape[1] != 8:
                continue
            dct_block = cv2.dct(block)
            for pos in coeff_positions:
                if abs(dct_block[pos]) > 1e-5:
                    coeff_magnitudes.append(abs(dct_block[pos]))

    threshold = np.median(coeff_magnitudes) * 0.1 if coeff_magnitudes else 0.0
    if threshold < 1e-5:
        threshold = 0.0

    bit_idx = 0

    rs = RSCodec(10)
    num_orig_bytes = (num_bits + 7) // 8
    num_rs_bytes = num_orig_bytes + 10
    num_encoded_bits = num_rs_bytes * 8
    num_final_bits = num_encoded_bits * repetition_factor

    max_possible_bits = (h//8) * (w//8) * len(coeff_positions)
    if num_final_bits > max_possible_bits:
        print(f"Warning: Not enough blocks in image. Truncating extraction.")
        num_final_bits = max_possible_bits

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if bit_idx >= num_final_bits:
                break
            block = image[i:i+8, j:j+8, channel_to_extract].astype(np.float32)
            if block.shape[0] != 8 or block.shape[1] != 8:
                continue

            dct_block = cv2.dct(block)

            for pos in coeff_positions:
                if bit_idx < num_final_bits:
                    bit = 1 if dct_block[pos] > threshold else 0
                    extracted_bits_repeated.append(bit)
                    bit_idx += 1
        if bit_idx >= num_final_bits:
            break

    decoded_bits = []
    for i in range(0, len(extracted_bits_repeated), repetition_factor):
        chunk = extracted_bits_repeated[i:i+repetition_factor]
        if len(chunk) == repetition_factor:
            ones = sum(chunk)
            decoded_bits.append(1 if ones > repetition_factor // 2 else 0)
        elif chunk:
            ones = sum(chunk)
            decoded_bits.append(1 if ones > len(chunk) // 2 else 0)

    decoded_bits = decoded_bits[:num_encoded_bits]

    num_bytes = (len(decoded_bits) + 7) // 8
    if num_bytes < num_rs_bytes:
        decoded_bits.extend([0] * ((num_rs_bytes - num_bytes) * 8))

    rs_bytes = bytearray([int(''.join(map(str, decoded_bits[i:i+8])), 2) for i in range(0, len(decoded_bits), 8)][:num_rs_bytes])

    try:
        rs_decoded, rs_decoded_erasures = rs.decode(rs_bytes)
        watermark_bits = [int(b) for byte in rs_decoded.decode('latin-1') for b in format(ord(byte), '08b')]
        return watermark_bits[:num_bits]
    except Exception as e:
        print(f"Reed-Solomon decoding error: {e}")
        return decoded_bits[:num_bits]

def bits_to_text(bits: List[int], expected_len_chars: int) -> str:
    chars = []
    for i in range(0, min(len(bits), expected_len_chars * 8), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        try:
            val = int(''.join(str(b) for b in byte), 2)
            if val < 256:
                chars.append(chr(val))
        except ValueError:
            pass
    return ''.join(chars)


def quantum_inspired_encoding(message: str, num_qubits: int = 4, noisy: bool = False):
    binary_message = ''.join([format(ord(c), '08b') for c in message])
    encoded_bits = ''
    chunk_size = num_qubits

    simulator = AerSimulator()
    if noisy:
        noise_model = NoiseModel()
        dep_error_1q = depolarizing_error(0.01, 1)
        dep_error_2q = depolarizing_error(0.01, 2)
        noise_model.add_all_qubit_quantum_error(dep_error_1q, ['h', 'x', 'measure'])
        noise_model.add_all_qubit_quantum_error(dep_error_2q, ['cx'])
        simulator = AerSimulator(noise_model=noise_model)

    for i in range(0, len(binary_message), chunk_size):
        qc = QuantumCircuit(num_qubits, num_qubits)
        chunk = binary_message[i:i+chunk_size].ljust(chunk_size, '0')

        qc.h(range(num_qubits))

        for j, bit in enumerate(chunk):
            if bit == '1':
                qc.x(j)

        for j in range(num_qubits - 1):
            qc.cx(j, j+1)

        qc.measure(range(num_qubits), range(num_qubits))

        compiled_circuit = transpile(qc, simulator)
        result = simulator.run(compiled_circuit, shots=1).result()
        counts = result.get_counts()
        measured = list(counts.keys())[0]

        measured_correct_order = measured[::-1]

        encoded_bits += measured_correct_order.zfill(chunk_size)[:chunk_size]

    encoded_message = ''
    padded = encoded_bits + ('0' * ((8 - len(encoded_bits) % 8) % 8))
    for m in range(0, len(padded), 8):
        try:
            encoded_message += chr(int(padded[m:m+8], 2))
        except ValueError:
            encoded_message += '?'

    return encoded_bits, encoded_message

def evaluate_image_quality(original_path: str, modified_path: str) -> Tuple[float, float]:
    orig_img = cv2.imread(original_path)
    mod_img = cv2.imread(modified_path)

    if orig_img is None or mod_img is None:
        try:
            orig_img = np.array(Image.open(original_path).convert('RGB'))
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
            mod_img = np.array(Image.open(modified_path).convert('RGB'))
            mod_img = cv2.cvtColor(mod_img, cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError(f"Failed to load images for quality evaluation: {original_path}, {modified_path}")

    if orig_img.shape != mod_img.shape:
        print(f"Warning: Mismatched shapes {orig_img.shape} vs {mod_img.shape}. Resizing modified image.")
        mod_img = cv2.resize(mod_img, (orig_img.shape[1], orig_img.shape[0]))

    psnr_value = psnr(orig_img, mod_img, data_range=255)
    if psnr_value == float('inf'):
        psnr_value = 100.0

    min_dim = min(orig_img.shape[0], orig_img.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)

    if win_size < 3:
        print(f"Warning: Image dimension {min_dim} is too small for default SSIM. Skipping SSIM.")
        ssim_value = 0.0
    else:
        ssim_value = ssim(orig_img, mod_img, data_range=255, multichannel=True, channel_axis=2, win_size=win_size)

    return psnr_value, ssim_value


def evaluate_more_metrics(original_path: str, modified_path: str) -> Tuple[float, float, float]:
    orig_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    mod_img = cv2.imread(modified_path, cv2.IMREAD_GRAYSCALE)

    if orig_img is None or mod_img is None:
        try:
            orig_img = np.array(Image.open(original_path).convert('L'))
            mod_img = np.array(Image.open(modified_path).convert('L'))
        except Exception:
            raise ValueError(f"Failed to load images for additional quality evaluation: {original_path}, {modified_path}")

    if orig_img.shape != mod_img.shape:
        print(f"Warning: Mismatched shapes {orig_img.shape} vs {mod_img.shape}. Resizing modified image for additional metrics.")
        mod_img = cv2.resize(mod_img, (orig_img.shape[1], orig_img.shape[0]))

    orig_flat = orig_img.flatten()
    mod_flat = mod_img.flatten()

    mae_value = mean_absolute_error(orig_flat, mod_flat)
    rmse_value = np.sqrt(mean_squared_error(orig_flat, mod_flat))

    a = orig_img.astype(float)
    b = mod_img.astype(float)
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    std_a = np.std(a)
    std_b = np.std(b)

    if std_a == 0 or std_b == 0:
        ncc_value = 0.0
    else:
        ncc_value = np.mean((a - mean_a) * (b - mean_b)) / (std_a * std_b)

    return mae_value, rmse_value, ncc_value


def apply_compression_attack(image_path: str, output_path: str, quality: int = 75):
    try:
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    except Exception as e:
        raise IOError(f"Failed to apply compression attack to {image_path}: {e}")

def calculate_ber(original_message: str, extracted_message: str) -> float:
    binary_orig = ''.join([format(ord(c), '08b') for c in original_message])
    binary_extr = ''.join([format(ord(c), '08b') for c in extracted_message])

    min_len = min(len(binary_orig), len(binary_extr))
    if min_len == 0:
        if len(binary_orig) > 0:
            return 1.0
        return 0.0

    max_len = max(len(binary_orig), len(binary_extr))
    binary_orig = binary_orig.ljust(max_len, '0')
    binary_extr = binary_extr.ljust(max_len, '0')

    errors = sum(1 for a, b in zip(binary_orig, binary_extr) if a != b)
    return errors / max_len

def bit_error_rate_bits(original_bits: List[int], extracted_bits: List[int]) -> float:
    min_len = min(len(original_bits), len(extracted_bits))
    if min_len == 0:
        if len(original_bits) > 0:
            return 1.0
        return 0.0

    max_len = max(len(original_bits), len(extracted_bits))
    orig_bits_padded = original_bits + [0] * (max_len - len(original_bits))
    extr_bits_padded = extracted_bits + [0] * (max_len - len(extracted_bits))

    errors = sum(1 for a, b in zip(orig_bits_padded, extr_bits_padded) if a != b)
    return errors / max_len


def main():
    secret_message = "SecretData2025"
    watermark_text = "OwnerID_Quantum"

    image_paths = load_images('images')
    results = {}
    os.makedirs('images/stego', exist_ok=True)
    os.makedirs('images/watermarked', exist_ok=True)
    os.makedirs('images/attacked', exist_ok=True)
    os.makedirs('images/plots', exist_ok=True)

    for idx, img_path in enumerate(image_paths):
        base = os.path.splitext(os.path.basename(img_path))[0]

        try:
            print(f"\n--- Processing {base} ---")

            stego_path = f"images/stego/{base}_stego.png"
            lsb_steganography(img_path, secret_message, stego_path)

            stego_arr = np.array(Image.open(stego_path).convert('RGB'), dtype=np.uint8)
            watermarked_arr, watermark_bits, encoded_bits = embed_watermark_dct_from_text(stego_arr, watermark_text, strength=75)
            wm_path = f"images/watermarked/{base}_wm.png"
            Image.fromarray(watermarked_arr).save(wm_path, 'PNG')

            print(f"Original watermark: {len(watermark_bits)} bits. RS+Rep-encoded: {len(encoded_bits)} bits.")

            encoded_bits_q, encoded_msg = quantum_inspired_encoding(secret_message, noisy=True)
            print(f"Quantum-inspired (first 80 bits): {encoded_bits_q[:80]}...")

            psnr_s, ssim_s = evaluate_image_quality(img_path, stego_path)
            mae_s, rmse_s, ncc_s = evaluate_more_metrics(img_path, stego_path)
            print(f"Stego vs Original: PSNR={psnr_s:.2f}dB, SSIM={ssim_s:.4f}, MAE={mae_s:.2f}, RMSE={rmse_s:.2f}, NCC={ncc_s:.4f}")

            psnr_w, ssim_w = evaluate_image_quality(img_path, wm_path)
            mae_w, rmse_w, ncc_w = evaluate_more_metrics(img_path, wm_path)
            print(f"WM vs Original:    PSNR={psnr_w:.2f}dB, SSIM={ssim_w:.4f}, MAE={mae_w:.2f}, RMSE={rmse_w:.2f}, NCC={ncc_w:.4f}")

            for quality in [70, 80, 90]:
                print(f"\n--- Testing JPEG quality {quality} for {base} ---")

                attacked_stego_jpeg = f"images/attacked/{base}_stego_q{quality}.jpg"
                apply_compression_attack(stego_path, attacked_stego_jpeg, quality=quality)

                try:
                    extracted_lsb_msg = extract_lsb_steganography(attacked_stego_jpeg, len(secret_message))
                    ber_msg = calculate_ber(secret_message, extracted_lsb_msg)
                except Exception as ex:
                    print(f"Error extracting LSB: {ex}")
                    extracted_lsb_msg = f"<extract error>"
                    ber_msg = 1.0
                print(f"LSB after JPEG (q={quality}): extracted = '{extracted_lsb_msg}', BER = {ber_msg:.4f}")

                attacked_wm_jpeg = f"images/attacked/{base}_wm_q{quality}.jpg"
                apply_compression_attack(wm_path, attacked_wm_jpeg, quality=quality)
                attacked_wm_arr = np.array(Image.open(attacked_wm_jpeg).convert('RGB'), dtype=np.uint8)

                extracted_wm_bits = extract_watermark_dct_to_bits(attacked_wm_arr, len(watermark_bits), repetition_factor=7)
                extracted_wm_text = bits_to_text(extracted_wm_bits, len(watermark_text))
                ber_wm = bit_error_rate_bits(watermark_bits, extracted_wm_bits)
                print(f"WM after JPEG (q={quality}): extracted = '{extracted_wm_text}', BER bits = {ber_wm:.4f}")

                results[f"{base}_q{quality}"] = {
                    'image_base': base,
                    'jpeg_quality': quality,
                    'psnr_stego': psnr_s,
                    'ssim_stego': ssim_s,
                    'mae_stego': mae_s,
                    'rmse_stego': rmse_s,
                    'ncc_stego': ncc_s,
                    'psnr_wm': psnr_w,
                    'ssim_wm': ssim_w,
                    'mae_wm': mae_w,
                    'rmse_wm': rmse_w,
                    'ncc_wm': ncc_w,
                    'ber_lsb_message_jpeg': ber_msg,
                    'extracted_lsb_message': extracted_lsb_msg,
                    'ber_wm_bits_jpeg': ber_wm,
                    'extracted_wm_text': extracted_wm_text,
                }

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 4, 1); plt.imshow(np.array(Image.open(img_path).convert('RGB'))); plt.title('Original'); plt.axis('off')
            plt.subplot(1, 4, 2); plt.imshow(np.array(Image.open(stego_path).convert('RGB'))); plt.title('Stego (LSB)'); plt.axis('off')
            plt.subplot(1, 4, 3); plt.imshow(watermarked_arr); plt.title('Stego + WM (DCT)'); plt.axis('off')
            plt.subplot(1, 4, 4); plt.imshow(np.array(Image.open(f"images/attacked/{base}_wm_q70.jpg").convert('RGB'))); plt.title('WM After JPEG q=70'); plt.axis('off')
            plt.suptitle(f'Image Processing Pipeline for {base}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'images/plots/{base}_comparison_stages.png')
            plt.show()

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            results[f"{base}_error"] = {
                'image_base': base, 'jpeg_quality': None, 'error': str(e)
            }

    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('summary_table.csv')
    print("\n--- Summary Table ---")
    print(df)
    print("\nSaved summary_table.csv")

    print("\nGenerating analysis graphs...")

    if df.empty or 'image_base' not in df.columns:
        print("DataFrame is empty or missing 'image_base' column. Skipping graphs.")
        return

    plt.figure(figsize=(10, 6))
    plt.title('BER vs. JPEG Compression Quality', fontsize=16)

    unique_images = df['image_base'].unique()

    for image_name in unique_images:
        subset = df[df['image_base'] == image_name].sort_values('jpeg_quality')
        if not subset.empty:
            plt.plot(subset['jpeg_quality'], subset['ber_lsb_message_jpeg'], marker='s', linestyle='--', label=f'{image_name} (LSB Message)')
            plt.plot(subset['jpeg_quality'], subset['ber_wm_bits_jpeg'], marker='o', linestyle='-', label=f'{image_name} (DCT Watermark)')

    plt.xlabel('JPEG Quality', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.xticks([70, 80, 90])
    plt.ylim(bottom=-0.05, top=1.05)
    plt.savefig('images/plots/ber_vs_jpeg_quality.png')
    print("Saved images/plots/ber_vs_jpeg_quality.png")
    plt.show()

    quality_df = df.drop_duplicates(subset=['image_base']).copy()

    if not quality_df.empty:
        bar_width = 0.2
        index = np.arange(len(quality_df['image_base']))

        fig, ax1 = plt.subplots(figsize=(14, 8))

        ax1.set_title('Image Quality After Embedding (All Metrics)', fontsize=16)
        ax1.set_xlabel('Image', fontsize=12)
        ax1.set_ylabel('PSNR (dB) / RMSE', fontsize=12, color='blue')

        ax1.bar(index - bar_width*1.5, quality_df['psnr_stego'], bar_width, label='PSNR (Stego)', color='blue', alpha=0.7)
        ax1.bar(index - bar_width*0.5, quality_df['psnr_wm'], bar_width, label='PSNR (Stego+WM)', color='cyan', alpha=0.7)
        ax1.bar(index + bar_width*0.5, quality_df['rmse_stego'], bar_width, label='RMSE (Stego)', color='purple', alpha=0.7)
        ax1.bar(index + bar_width*1.5, quality_df['rmse_wm'], bar_width, label='RMSE (Stego+WM)', color='pink', alpha=0.7)

        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xticks(index)
        ax1.set_xticklabels(quality_df['image_base'])

        ax2 = ax1.twinx()
        ax2.set_ylabel('SSIM / NCC / MAE', fontsize=12, color='red')

        ax2.plot(index, quality_df['ssim_stego'], marker='o', linestyle='none', markersize=10, label='SSIM (Stego)', color='red')
        ax2.plot(index, quality_df['ssim_wm'], marker='D', linestyle='none', markersize=10, label='SSIM (Stego+WM)', color='orange')
        ax2.plot(index, quality_df['mae_stego'], marker='^', linestyle='none', markersize=10, label='MAE (Stego)', color='green')
        ax2.plot(index, quality_df['mae_wm'], marker='v', linestyle='none', markersize=10, label='MAE (Stego+WM)', color='lime')
        ax2.plot(index, quality_df['ncc_stego'], marker='s', linestyle='none', markersize=10, label='NCC (Stego)', color='brown')
        ax2.plot(index, quality_df['ncc_wm'], marker='p', linestyle='none', markersize=10, label='NCC (Stego+WM)', color='gray')

        ax2.tick_params(axis='y', labelcolor='red')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=6, fontsize=10)

        fig.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig('images/plots/embedding_quality_all_metrics.png')
        print("Saved images/plots/embedding_quality_all_metrics.png")
        plt.show()

    print("Generating grouped bar chart for BER comparison...")
    try:
        df_q70 = df[df['jpeg_quality'] == 70].copy()

        if df_q70.empty:
            print("No data for JPEG quality 70. Skipping BER plot 3.")
        else:
            labels = df_q70['image_base']
            ber_lsb = df_q70['ber_lsb_message_jpeg']
            ber_dct = df_q70['ber_wm_bits_jpeg']

            x = np.arange(len(labels))
            width = 0.35

            fig, ax = plt.subplots(figsize=(12, 7))
            rects1 = ax.bar(x - width/2, ber_lsb, width, label='BER LSB (Fragile)', color='tomato')
            rects2 = ax.bar(x + width/2, ber_dct, width, label='BER DCT (Robust)', color='dodgerblue')

            ax.set_ylabel('Bit Error Rate (BER)')
            ax.set_title('Robustness Comparison at Harshest JPEG (q=70)')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1.1)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            ax.bar_label(rects1, padding=3, fmt='%.2f')
            ax.bar_label(rects2, padding=3, fmt='%.2f')

            fig.tight_layout()
            plt.savefig('images/plots/ber_comparison_q70_barchart.png')
            print("Saved images/plots/ber_comparison_q70_barchart.png")
            plt.show()

    except Exception as e:
        print(f"Error generating Plot 3: {e}")

    print("Generating new grouped bar chart for PSNR comparison...")
    try:
        if quality_df.empty:
            print("No data for plotting. Skipping PSNR plot 4.")
        else:
            labels = quality_df['image_base']
            psnr_lsb = quality_df['psnr_stego']
            psnr_dct = quality_df['psnr_wm']

            x = np.arange(len(labels))
            width = 0.35

            fig, ax = plt.subplots(figsize=(12, 7))
            rects1 = ax.bar(x - width/2, psnr_lsb, width, label='PSNR (LSB Stego Only)', color='mediumseagreen')
            rects2 = ax.bar(x + width/2, psnr_dct, width, label='PSNR (LSB + DCT WM)', color='darkorange')

            ax.set_ylabel('PSNR (dB) - Higher is Better')
            ax.set_title('Imperceptibility Cost: PSNR Comparison After Embedding')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)

            all_psnr_values = pd.concat([psnr_lsb, psnr_dct])
            min_psnr = all_psnr_values.min()
            ax.set_ylim(bottom=max(0, min_psnr - 5))

            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            ax.bar_label(rects1, padding=3, fmt='%.2f')
            ax.bar_label(rects2, padding=3, fmt='%.2f')

            fig.tight_layout()
            plt.savefig('images/plots/psnr_cost_comparison_barchart.png')
            print("Saved images/plots/psnr_cost_comparison_barchart.png")
            plt.show()

    except Exception as e:
        print(f"Error generating Plot 4 (NEW): {e}")

    print("Generating new grouped bar chart for SSIM comparison...")
    try:
        if quality_df.empty:
            print("No data for plotting. Skipping SSIM plot 5.")
        else:
            labels = quality_df['image_base']
            ssim_lsb = quality_df['ssim_stego']
            ssim_dct = quality_df['ssim_wm']

            x = np.arange(len(labels))
            width = 0.35

            fig, ax = plt.subplots(figsize=(12, 7))
            rects1 = ax.bar(x - width/2, ssim_lsb, width, label='SSIM (LSB Stego Only)', color='cornflowerblue')
            rects2 = ax.bar(x + width/2, ssim_dct, width, label='SSIM (LSB + DCT WM)', color='palevioletred')

            ax.set_ylabel('SSIM - Higher is Better (Max 1.0)')
            ax.set_title('Imperceptibility Cost: SSIM Comparison After Embedding')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)

            all_ssim_values = pd.concat([ssim_lsb, ssim_dct])
            min_ssim = all_ssim_values.min()
            ax.set_ylim(bottom=max(0.85, min_ssim - 0.02), top=1.01)

            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            ax.bar_label(rects1, padding=3, fmt='%.4f')
            ax.bar_label(rects2, padding=3, fmt='%.4f')

            fig.tight_layout()
            plt.savefig('images/plots/ssim_cost_comparison_barchart.png')
            print("Saved images/plots/ssim_cost_comparison_barchart.png")
            plt.show()

    except Exception as e:
        print(f"Error generating Plot 5 (NEW): {e}")


if __name__ == '__main__':
    main()
