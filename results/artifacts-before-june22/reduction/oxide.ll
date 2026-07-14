; ModuleID = 'builtin.module'
source_filename = "reduction"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_0 = addrspace(3) global [256 x float] zeroinitializer, align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare void @llvm.nvvm.barrier0() #0

define ptx_kernel void @reduce(ptr %v0, i64 %v1, ptr %v2, i64 %v3) {
entry:
  %v4 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64 } %v4, i64 %v1, 1
  %v6 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v3, 1
  br label %bb0
bb0:
  %v8 = phi { ptr, i64 } [ %v5, %entry ]
  %v9 = phi { ptr, i64 } [ %v7, %entry ]
  %v10 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb1
bb1:
  %v11 = zext i32 %v10 to i64
  %v12 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb2
bb2:
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb3
bb3:
  %v14 = mul i32 %v12, %v13
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb4
bb4:
  %v16 = add i32 %v14, %v15
  %v17 = zext i32 %v16 to i64
  %v18 = extractvalue { ptr, i64 } %v8, 1
  %v19 = icmp ult i64 %v17, %v18
  %v20 = xor i1 %v19, 1
  br i1 %v20, label %bb6, label %bb5
bb5:
  %v21 = extractvalue { ptr, i64 } %v8, 0
  %v22 = getelementptr inbounds float, ptr %v21, i64 %v17
  %v23 = load float, ptr %v22
  br label %bb7
bb6:
  br label %bb7
bb7:
  %v24 = phi float [ %v23, %bb5 ], [ 0.0, %bb6 ]
  %v25 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_0, i64 %v11
  br label %bb8
bb8:
  store float %v24, ptr addrspace(3) %v25
  call void @llvm.nvvm.barrier0() #0
  br label %bb9
bb9:
  %v27 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb10
bb10:
  %v28 = udiv i32 %v27, 2
  %v29 = zext i32 %v28 to i64
  br label %bb11
bb11:
  %v30 = phi i64 [ %v29, %bb10 ], [ %v47, %bb19 ]
  %v31 = icmp ugt i64 %v30, 0
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb20, label %bb12
bb12:
  %v33 = icmp ult i64 %v11, %v30
  %v34 = xor i1 %v33, 1
  br i1 %v34, label %bb17, label %bb13
bb13:
  %v35 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v36 = getelementptr inbounds float, ptr addrspace(3) %v35, i64 %v11
  br label %bb14
bb14:
  %v37 = load float, ptr addrspace(3) %v36
  %v38 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v39 = add i64 %v11, %v30
  %v40 = getelementptr inbounds float, ptr addrspace(3) %v38, i64 %v39
  br label %bb15
bb15:
  %v41 = load float, ptr addrspace(3) %v40
  %v42 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_0, i64 %v11
  br label %bb16
bb16:
  %v43 = fadd float %v37, %v41
  store float %v43, ptr addrspace(3) %v42
  br label %bb18
bb17:
  br label %bb18
bb18:
  call void @llvm.nvvm.barrier0() #0
  br label %bb19
bb19:
  %v45 = zext i32 1 to i64
  %v46 = and i64 %v45, 63
  %v47 = lshr i64 %v30, %v46
  br label %bb11
bb20:
  %v48 = icmp eq i64 %v11, 0
  br i1 %v48, label %bb21, label %bb24
bb21:
  %v49 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb22
bb22:
  %v50 = zext i32 %v49 to i64
  %v51 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v52 = getelementptr inbounds float, ptr addrspace(3) %v51, i64 0
  br label %bb23
bb23:
  %v53 = load float, ptr addrspace(3) %v52
  %v54 = extractvalue { ptr, i64 } %v9, 0
  %v55 = getelementptr inbounds float, ptr %v54, i64 %v50
  store float %v53, ptr %v55
  br label %bb24
bb24:
  ret void
}


attributes #0 = { convergent }
