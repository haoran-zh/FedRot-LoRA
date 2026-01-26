import torch
import logging
from collections import OrderedDict

from federatedscope.core.auxiliaries.utils import get_ds_rank

logger = logging.getLogger(__name__)
if get_ds_rank() == 0:
    logger.setLevel(logging.INFO)

def rotation_align_optimization(initial_model_ref: torch.Tensor,
                                align_matrix: str,
                                updated_A: torch.Tensor,
                                updated_B: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Performs rotation alignment for a *single pair* of LoRA parameters using PyTorch.

    This function solves the Orthogonal Procrustes problem to find the
    optimal rotation matrix (R) that aligns a client's updated parameters
    to a shared reference (the initial_model_ref). It then applies this
    rotation to both A and B.

    Args:
        initial_model_ref (torch.Tensor): The reference matrix (on the same device as updated_A/B).
        align_matrix (str): Either 'A' or 'B'. Specifies which matrix to use
            for the alignment calculation.
        updated_A (torch.Tensor): The client's locally updated A matrix (r x d).
        updated_B (torch.Tensor): The client's locally updated B matrix (d x r).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - rotated_A (torch.Tensor): The new A matrix (R.T @ A).
            - rotated_B (torch.Tensor): The new B matrix (B @ R).
    """

    # Get original dtype and device to ensure consistency
    # We'll use the dtypes of A and B directly for casting.
    a_dtype = updated_A.dtype
    b_dtype = updated_B.dtype
    original_device = updated_A.device

    # 1. Find the optimal rotation R by solving the Orthogonal Procrustes problem.
    # We want to find an r x r rotation matrix R.
    # All computations will stay on the device of the input tensors (e.g., 'cuda:0').

    with torch.no_grad():  # Ensure no gradients are computed during alignment
        if align_matrix == 'A':
            # M = updated_A @ initial_model_ref.T
            # Ensure consistent dtype for matmul, upcasting to float32 if dtypes differ
            M = torch.matmul(updated_A.to(torch.float32), initial_model_ref.T.to(torch.float32))
        elif align_matrix == 'B':
            # M = updated_B.T @ initial_model_ref
            # Ensure consistent dtype for matmul, upcasting to float32 if dtypes differ
            M = torch.matmul(updated_B.T.to(torch.float32), initial_model_ref.to(torch.float32))
        else:
            raise ValueError("align_matrix must be 'A' or 'B'")

        # 2. Compute SVD of the r x r correlation matrix M using torch.linalg.svd
        try:
            # --- FIX for 'Half' precision error ---
            # SVD on CUDA does not support float16/bfloat16. M is already float32.
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        except torch.linalg.LinAlgError:
            print(
                f"Warning: SVD computation failed for matrix of shape {M.shape} on device {M.device}. Using identity matrix.")
            # Fallback to a simple identity matrix on the same device
            R_32 = torch.eye(M.shape[0], device=original_device, dtype=torch.float32)
        else:
            # The optimal rotation matrix R = U @ Vh
            # R is computed in float32.
            R_32 = torch.matmul(U, Vh)

        # 3. Apply the rotation to both A and B
        # B' = B @ R
        # A' = R.T @ A

        # --- FIX for dtype mismatch ---
        # Explicitly cast R (which is float32) to the *specific* dtype
        # of the matrix it is being multiplied with.

        # Cast R to B's dtype for the first multiplication
        rotated_B = torch.matmul(updated_B, R_32.to(b_dtype))

        # Cast R.T to A's dtype for the second multiplication
        rotated_A = torch.matmul(R_32.T.to(a_dtype), updated_A)

    return rotated_A, rotated_B


def rotation_align_optimization_soft(initial_model_ref: torch.Tensor,
                                align_matrix: str,
                                updated_A: torch.Tensor,
                                updated_B: torch.Tensor,
                                rotation_lambda) -> (torch.Tensor, torch.Tensor):
    """
    Performs rotation alignment for a *single pair* of LoRA parameters using PyTorch.

    This function solves the Orthogonal Procrustes problem to find the
    optimal rotation matrix (R) that aligns a client's updated parameters
    to a shared reference (the initial_model_ref). It then applies this
    rotation to both A and B.

    Args:
        initial_model_ref (torch.Tensor): The reference matrix (on the same device as updated_A/B).
        align_matrix (str): Either 'A' or 'B'. Specifies which matrix to use
            for the alignment calculation.
        updated_A (torch.Tensor): The client's locally updated A matrix (r x d).
        updated_B (torch.Tensor): The client's locally updated B matrix (d x r).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - rotated_A (torch.Tensor): The new A matrix (R.T @ A).
            - rotated_B (torch.Tensor): The new B matrix (B @ R).
    """

    # Get original dtype and device to ensure consistency
    # We'll use the dtypes of A and B directly for casting.
    a_dtype = updated_A.dtype
    b_dtype = updated_B.dtype
    original_device = updated_A.device

    # 1. Find the optimal rotation R by solving the Orthogonal Procrustes problem.
    # We want to find an r x r rotation matrix R.
    # All computations will stay on the device of the input tensors (e.g., 'cuda:0').

    with torch.no_grad():  # Ensure no gradients are computed during alignment
        if align_matrix == 'A':
            # M = updated_A @ initial_model_ref.T
            # Ensure consistent dtype for matmul, upcasting to float32 if dtypes differ
            M = torch.matmul(updated_A.to(torch.float32), initial_model_ref.T.to(torch.float32))
        elif align_matrix == 'B':
            # M = updated_B.T @ initial_model_ref
            # Ensure consistent dtype for matmul, upcasting to float32 if dtypes differ
            M = torch.matmul(updated_B.T.to(torch.float32), initial_model_ref.to(torch.float32))
        else:
            raise ValueError("align_matrix must be 'A' or 'B'")

        # 2. Compute SVD of the r x r correlation matrix M using torch.linalg.svd
        try:
            # --- FIX for 'Half' precision error ---
            # SVD on CUDA does not support float16/bfloat16. M is already float32.
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        except torch.linalg.LinAlgError:
            print(
                f"Warning: SVD computation failed for matrix of shape {M.shape} on device {M.device}. Using identity matrix.")
            # Fallback to a simple identity matrix on the same device
            R_32 = torch.eye(M.shape[0], device=original_device, dtype=torch.float32)
        else:
            # The optimal rotation matrix R = U @ Vh
            # R is computed in float32.
            R_32 = torch.matmul(U, Vh)

        # If determinant is -1, SVD gave us a reflection (mirror), not a rotation.
        # This causes catastrophic cancellation. We must flip it back.
        if torch.linalg.det(R_32) < 0:
            # Flip the sign of the last row of Vh (or last col of U)
            Vh[-1, :] *= -1
            R_32 = torch.matmul(U, Vh)
            print('need to flip for rotation matrix')
        if rotation_lambda == 1.0:
            pass
        else:
            # Linear Interpolation: (1 - lambda) * I + lambda * R_target
            I = torch.eye(R_32.shape[0], device=original_device, dtype=torch.float32)
            R_soft = (1 - rotation_lambda) * I + rotation_lambda * R_32

            # Re-orthogonalize: Project R_soft back to the closest orthogonal matrix
            # We do this by taking the SVD of the interpolated matrix: R_soft = U' S' V'T
            # The closest orthogonal matrix is U' V'T
            try:
                U_s, _, Vh_s = torch.linalg.svd(R_soft, full_matrices=False)
                R_32 = torch.matmul(U_s, Vh_s)
            except torch.linalg.LinAlgError:
                # Fallback to hard rotation if re-orthogenalization fails
                pass

        # Cast R to B's dtype for the first multiplication
        rotated_B = torch.matmul(updated_B, R_32.to(b_dtype))

        # Cast R.T to A's dtype for the second multiplication
        rotated_A = torch.matmul(R_32.T.to(a_dtype), updated_A)

    return rotated_A, rotated_B




def rotation_align_optimization_regularized(initial_model_ref: torch.Tensor,
                                align_matrix: str,
                                updated_A: torch.Tensor,
                                updated_B: torch.Tensor,
                                rotation_lambda) -> (torch.Tensor, torch.Tensor):
    """
    Performs rotation alignment for a *single pair* of LoRA parameters using PyTorch.

    This function solves the Orthogonal Procrustes problem to find the
    optimal rotation matrix (R) that aligns a client's updated parameters
    to a shared reference (the initial_model_ref). It then applies this
    rotation to both A and B.

    Args:
        initial_model_ref (torch.Tensor): The reference matrix (on the same device as updated_A/B).
        align_matrix (str): Either 'A' or 'B'. Specifies which matrix to use
            for the alignment calculation.
        updated_A (torch.Tensor): The client's locally updated A matrix (r x d).
        updated_B (torch.Tensor): The client's locally updated B matrix (d x r).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - rotated_A (torch.Tensor): The new A matrix (R.T @ A).
            - rotated_B (torch.Tensor): The new B matrix (B @ R).
    """

    # Get original dtype and device to ensure consistency
    # We'll use the dtypes of A and B directly for casting.
    a_dtype = updated_A.dtype
    b_dtype = updated_B.dtype
    original_device = updated_A.device

    # 1. Find the optimal rotation R by solving the Orthogonal Procrustes problem.
    # We want to find an r x r rotation matrix R.
    # All computations will stay on the device of the input tensors (e.g., 'cuda:0').

    with torch.no_grad():  # Ensure no gradients are computed during alignment
        if align_matrix == 'A':
            # M = updated_A @ initial_model_ref.T
            # Ensure consistent dtype for matmul, upcasting to float32 if dtypes differ
            M = torch.matmul(updated_A.to(torch.float32), initial_model_ref.T.to(torch.float32))
        elif align_matrix == 'B':
            # M = updated_B.T @ initial_model_ref
            # Ensure consistent dtype for matmul, upcasting to float32 if dtypes differ
            M = torch.matmul(updated_B.T.to(torch.float32), initial_model_ref.to(torch.float32))
        else:
            raise ValueError("align_matrix must be 'A' or 'B'")

        if rotation_lambda > 0:
            I = torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
            M = M + (rotation_lambda * I)

        # 2. Compute SVD of the r x r correlation matrix M using torch.linalg.svd
        try:
            # --- FIX for 'Half' precision error ---
            # SVD on CUDA does not support float16/bfloat16. M is already float32.
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        except torch.linalg.LinAlgError:
            print(
                f"Warning: SVD computation failed for matrix of shape {M.shape} on device {M.device}. Using identity matrix.")
            # Fallback to a simple identity matrix on the same device
            R_32 = torch.eye(M.shape[0], device=original_device, dtype=torch.float32)
        else:
            # The optimal rotation matrix R = U @ Vh
            # R is computed in float32.
            R_32 = torch.matmul(U, Vh)

        # If determinant is -1, SVD gave us a reflection (mirror), not a rotation.
        # This causes catastrophic cancellation. We must flip it back.
        if torch.linalg.det(R_32) < 0:
            # Flip the sign of the last row of Vh (or last col of U)
            Vh[-1, :] *= -1
            R_32 = torch.matmul(U, Vh)
            print('need to flip for rotation matrix')

        # Cast R to B's dtype for the first multiplication
        rotated_B = torch.matmul(updated_B, R_32.to(b_dtype))

        # Cast R.T to A's dtype for the second multiplication
        rotated_A = torch.matmul(R_32.T.to(a_dtype), updated_A)

    return rotated_A, rotated_B


def rotation_alignment(initial_model_ref: dict,
                       align: str,
                       updated_A: dict,
                       updated_B: dict) -> (dict, dict):
    """
    Wrapper function to align an entire LoRA model (represented as dictionaries).

    This iterates through layers and calls the torch-based optimization.

    Args:
        initial_model_ref (dict): The reference model weights (e.g., A_ref_dict or B_ref_dict).
        align (str): Either 'A' or 'B'.
        updated_A (dict): Client's updated A matrices.
        updated_B (dict): Client's updated B matrices.

    Returns:
        tuple[dict, dict]: A tuple containing:
            - rotatedA (dict): The new, aligned A matrices.
            - rotatedB (dict): The new, aligned B matrices.
    """
    rotatedA = OrderedDict()
    rotatedB = OrderedDict()
    layer_keys_A = list(updated_A.keys())
    # Create corresponding B keys (e.g., 'layer1.lora_A' -> 'layer1.lora_B')
    layer_keys_B = [key.replace('.lora_A', '.lora_B') for key in layer_keys_A]  # ensure order

    if align == 'A':
        ref_keys = layer_keys_A
    elif align == 'B':
        ref_keys = layer_keys_B
    else:
        raise ValueError("align must be 'A' or 'B'")

    for key_a, key_b, key_ref in zip(layer_keys_A, layer_keys_B, ref_keys):
        # Check if the keys exist before processing
        if key_a not in updated_A or key_b not in updated_B or key_ref not in initial_model_ref:
            print(f"Warning: Skipping layer due to missing keys. Looked for {key_a}, {key_b}, {key_ref}")
            continue

        A = updated_A[key_a]
        B = updated_B[key_b]
        model_ref = initial_model_ref[key_ref]

        # Call the pure PyTorch function
        rotatedA[key_a], rotatedB[key_b] = rotation_align_optimization(model_ref, align, A, B)

    return rotatedA, rotatedB


def rotation_alignment_soft(initial_model_ref: dict,
                       align: str,
                       updated_A: dict,
                       updated_B: dict,
                        rotation_lambda: float) -> (dict, dict):
    # soft rotation alignment
    rotatedA = OrderedDict()
    rotatedB = OrderedDict()
    layer_keys_A = list(updated_A.keys())
    # Create corresponding B keys (e.g., 'layer1.lora_A' -> 'layer1.lora_B')
    layer_keys_B = [key.replace('.lora_A', '.lora_B') for key in layer_keys_A]  # ensure order

    if align == 'A':
        ref_keys = layer_keys_A
    elif align == 'B':
        ref_keys = layer_keys_B
    else:
        raise ValueError("align must be 'A' or 'B'")

    total_norm_A = 0.0
    total_norm_B = 0.0
    total_norm_Aprime= 0.0
    total_norm_Bprime= 0.0
    count = 0

    for key_a, key_b, key_ref in zip(layer_keys_A, layer_keys_B, ref_keys):
        # Check if the keys exist before processing
        if key_a not in updated_A or key_b not in updated_B or key_ref not in initial_model_ref:
            print(f"Warning: Skipping layer due to missing keys. Looked for {key_a}, {key_b}, {key_ref}")
            continue

        A = updated_A[key_a]
        B = updated_B[key_b]
        model_ref = initial_model_ref[key_ref]

        total_norm_A += torch.norm(A).item()
        total_norm_B += torch.norm(B).item()
        count += 1

        # Call the pure PyTorch function
        rotatedA[key_a], rotatedB[key_b] = rotation_align_optimization_soft(model_ref, align, A, B, rotation_lambda)

        # compute norms after rotation
        total_norm_Aprime += torch.norm(rotatedA[key_a]).item()
        total_norm_Bprime += torch.norm(rotatedB[key_b]).item()

    avg_norm_A = total_norm_A / count if count > 0 else 0.0
    avg_norm_B = total_norm_B / count if count > 0 else 0.0
    avg_norm_Aprime = total_norm_Aprime / count if count > 0 else 0.0
    avg_norm_Bprime = total_norm_Bprime / count if count > 0 else 0.0
    # print(f'Average norm of updated A ,B before alignment: A {avg_norm_A}, B {avg_norm_B}')
    # print(f'Average norm of updated A, B after alignment: A {avg_norm_Aprime}, B {avg_norm_Bprime}')
    logger.info(f'Average norm of updated A ,B before alignment: A {avg_norm_A}, B {avg_norm_B}')
    logger.info(f'Average norm of updated A, B after alignment: A {avg_norm_Aprime}, B {avg_norm_Bprime}')

    return rotatedA, rotatedB


def rotation_alignment_soft_normalized(initial_model_ref: dict,
                       align: str,
                       updated_A: dict,
                       updated_B: dict,
                        rotation_lambda: float) -> (dict, dict):
    # soft rotation alignment
    rotatedA = OrderedDict()
    rotatedB = OrderedDict()
    layer_keys_A = list(updated_A.keys())
    # Create corresponding B keys (e.g., 'layer1.lora_A' -> 'layer1.lora_B')
    layer_keys_B = [key.replace('.lora_A', '.lora_B') for key in layer_keys_A]  # ensure order

    if align == 'A':
        ref_keys = layer_keys_A
    elif align == 'B':
        ref_keys = layer_keys_B
    else:
        raise ValueError("align must be 'A' or 'B'")

    total_norm_A = 0.0
    total_norm_B = 0.0
    total_norm_Aprime= 0.0
    total_norm_Bprime= 0.0
    count = 0

    for key_a, key_b, key_ref in zip(layer_keys_A, layer_keys_B, ref_keys):
        # Check if the keys exist before processing
        if key_a not in updated_A or key_b not in updated_B or key_ref not in initial_model_ref:
            print(f"Warning: Skipping layer due to missing keys. Looked for {key_a}, {key_b}, {key_ref}")
            continue

        A = updated_A[key_a]
        B = updated_B[key_b]
        model_ref = initial_model_ref[key_ref]

        total_norm_A += torch.norm(A).item()
        total_norm_B += torch.norm(B).item()
        count += 1

        # normalized before rotation

        A_normalized = A / torch.norm(A)
        B_normalized = B / torch.norm(B)
        model_ref_normalized = model_ref / torch.norm(model_ref)


        # Call the pure PyTorch function
        rotatedA[key_a], rotatedB[key_b] = rotation_align_optimization_soft(model_ref_normalized, align, A_normalized, B_normalized, rotation_lambda)

        # recover original scale after rotation
        rotatedA[key_a] = rotatedA[key_a] * torch.norm(A)
        rotatedB[key_b] = rotatedB[key_b] * torch.norm(B)

        # compute norms after rotation
        total_norm_Aprime += torch.norm(rotatedA[key_a]).item()
        total_norm_Bprime += torch.norm(rotatedB[key_b]).item()

    avg_norm_A = total_norm_A / count if count > 0 else 0.0
    avg_norm_B = total_norm_B / count if count > 0 else 0.0
    avg_norm_Aprime = total_norm_Aprime / count if count > 0 else 0.0
    avg_norm_Bprime = total_norm_Bprime / count if count > 0 else 0.0
    logger.info(f'Average norm of updated A ,B before alignment: A {avg_norm_A}, B {avg_norm_B}')
    logger.info(f'Average norm of updated A, B after alignment: A {avg_norm_Aprime}, B {avg_norm_Bprime}')

    return rotatedA, rotatedB



def rotation_alignment_regularized(initial_model_ref: dict,
                       align: str,
                       updated_A: dict,
                       updated_B: dict,
                        rotation_lambda: float) -> (dict, dict):
    # soft rotation alignment
    rotatedA = OrderedDict()
    rotatedB = OrderedDict()
    layer_keys_A = list(updated_A.keys())
    # Create corresponding B keys (e.g., 'layer1.lora_A' -> 'layer1.lora_B')
    layer_keys_B = [key.replace('.lora_A', '.lora_B') for key in layer_keys_A]  # ensure order

    if align == 'A':
        ref_keys = layer_keys_A
    elif align == 'B':
        ref_keys = layer_keys_B
    else:
        raise ValueError("align must be 'A' or 'B'")

    total_norm_A = 0.0
    total_norm_B = 0.0
    total_norm_Aprime= 0.0
    total_norm_Bprime= 0.0
    count = 0

    for key_a, key_b, key_ref in zip(layer_keys_A, layer_keys_B, ref_keys):
        # Check if the keys exist before processing
        if key_a not in updated_A or key_b not in updated_B or key_ref not in initial_model_ref:
            print(f"Warning: Skipping layer due to missing keys. Looked for {key_a}, {key_b}, {key_ref}")
            continue

        A = updated_A[key_a]
        B = updated_B[key_b]
        model_ref = initial_model_ref[key_ref]

        total_norm_A += torch.norm(A).item()
        total_norm_B += torch.norm(B).item()
        count += 1

        # Call the pure PyTorch function
        rotatedA[key_a], rotatedB[key_b] = rotation_align_optimization_regularized(model_ref, align, A, B, rotation_lambda)

        # compute norms after rotation
        total_norm_Aprime += torch.norm(rotatedA[key_a]).item()
        total_norm_Bprime += torch.norm(rotatedB[key_b]).item()

    avg_norm_A = total_norm_A / count if count > 0 else 0.0
    avg_norm_B = total_norm_B / count if count > 0 else 0.0
    avg_norm_Aprime = total_norm_Aprime / count if count > 0 else 0.0
    avg_norm_Bprime = total_norm_Bprime / count if count > 0 else 0.0
    print(f'Average norm of updated A ,B before alignment: A {avg_norm_A}, B {avg_norm_B}')
    print(f'Average norm of updated A, B after alignment: A {avg_norm_Aprime}, B {avg_norm_Bprime}')
    logger.info(f'Average norm of updated A ,B before alignment: A {avg_norm_A}, B {avg_norm_B}')
    logger.info(f'Average norm of updated A, B after alignment: A {avg_norm_Aprime}, B {avg_norm_Bprime}')

    return rotatedA, rotatedB