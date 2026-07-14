; ModuleID = 'builtin.module'
source_filename = "gather"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @gather(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5) {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  %v10 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v5, 1
  br label %bb0
bb0:
  %v12 = phi { ptr, i64 } [ %v7, %entry ]
  %v13 = phi { ptr, i64 } [ %v9, %entry ]
  %v14 = phi { ptr, i64 } [ %v11, %entry ]
  %v15 = alloca {  }
  %v16 = bitcast ptr %v15 to ptr
  %v17 = call i64 @cuda_device____internal__index_1d(ptr %v16)
  br label %bb1
bb1:
  %v18 = extractvalue { ptr, i64 } %v14, 1
  %v19 = icmp ult i64 %v17, %v18
  %v20 = xor i1 %v19, 1
  br i1 %v20, label %bb8, label %bb7
bb2:
  %v21 = extractvalue { i8, ptr } %v38, 1
  %v22 = extractvalue { ptr, i64 } %v13, 1
  %v23 = icmp ult i64 %v17, %v22
  br i1 %v23, label %bb3, label %bb12
bb3:
  %v24 = extractvalue { ptr, i64 } %v13, 0
  %v25 = getelementptr inbounds i32, ptr %v24, i64 %v17
  %v26 = load i32, ptr %v25
  %v27 = zext i32 %v26 to i64
  %v28 = extractvalue { ptr, i64 } %v12, 1
  %v29 = icmp ult i64 %v27, %v28
  br i1 %v29, label %bb4, label %bb13
bb4:
  %v30 = extractvalue { ptr, i64 } %v12, 0
  %v31 = getelementptr inbounds float, ptr %v30, i64 %v27
  %v32 = load float, ptr %v31
  store float %v32, ptr %v21
  br label %bb6
bb5:
  br label %bb6
bb6:
  ret void
bb7:
  %v33 = extractvalue { ptr, i64 } %v14, 0
  %v34 = getelementptr inbounds float, ptr %v33, i64 %v17
  %v35 = insertvalue { i8, ptr } undef, i8 1, 0
  %v36 = insertvalue { i8, ptr } %v35, ptr %v34, 1
  br label %bb9
bb8:
  %v37 = insertvalue { i8, ptr } undef, i8 0, 0
  br label %bb9
bb9:
  %v38 = phi { i8, ptr } [ %v36, %bb7 ], [ %v37, %bb8 ]
  %v39 = extractvalue { i8, ptr } %v38, 0
  %v40 = zext i8 %v39 to i64
  %v41 = icmp eq i64 %v40, 1
  br i1 %v41, label %bb2, label %bb10
bb10:
  %v42 = icmp eq i64 %v40, 0
  br i1 %v42, label %bb5, label %bb11
bb11:
  unreachable
bb12:
  unreachable
bb13:
  unreachable
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

define i64 @cuda_device____internal__index_1d(ptr %v0) {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb1
bb1:
  %v3 = zext i32 %v2 to i64
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb2
bb2:
  %v5 = zext i32 %v4 to i64
  %v6 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb3
bb3:
  %v7 = zext i32 %v6 to i64
  %v8 = mul i64 %v5, %v7
  %v9 = add i64 %v8, %v3
  ret i64 %v9
}

