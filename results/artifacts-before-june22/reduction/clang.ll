; ModuleID = 'kernels/reduction/cuda/reduction.cu'
source_filename = "kernels/reduction/cuda/reduction.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@_ZZ6reduceE4smem = internal unnamed_addr addrspace(3) global [256 x float] undef, align 4

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @reduce(ptr noundef readonly captures(none) %0, ptr noundef writeonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %6 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %7 = mul i32 %5, %6
  %8 = add i32 %7, %4
  %9 = icmp ult i32 %8, %2
  br i1 %9, label %10, label %14

10:                                               ; preds = %3
  %11 = zext i32 %8 to i64
  %12 = getelementptr inbounds nuw float, ptr %0, i64 %11
  %13 = load float, ptr %12, align 4, !tbaa !7
  br label %14

14:                                               ; preds = %3, %10
  %15 = phi contract float [ %13, %10 ], [ 0.000000e+00, %3 ]
  %16 = zext nneg i32 %4 to i64
  %17 = getelementptr inbounds nuw [256 x float], ptr addrspacecast (ptr addrspace(3) @_ZZ6reduceE4smem to ptr), i64 0, i64 %16
  store float %15, ptr %17, align 4, !tbaa !7
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %18 = icmp samesign ult i32 %6, 2
  br i1 %18, label %19, label %21

19:                                               ; preds = %32, %14
  %20 = icmp eq i32 %4, 0
  br i1 %20, label %34, label %38

21:                                               ; preds = %14, %32
  %22 = phi i32 [ %23, %32 ], [ %6, %14 ]
  %23 = lshr i32 %22, 1
  %24 = icmp samesign ult i32 %4, %23
  br i1 %24, label %25, label %32

25:                                               ; preds = %21
  %26 = add nuw nsw i32 %23, %4
  %27 = zext nneg i32 %26 to i64
  %28 = getelementptr inbounds nuw [256 x float], ptr addrspacecast (ptr addrspace(3) @_ZZ6reduceE4smem to ptr), i64 0, i64 %27
  %29 = load float, ptr %28, align 4, !tbaa !7
  %30 = load float, ptr %17, align 4, !tbaa !7
  %31 = fadd contract float %29, %30
  store float %31, ptr %17, align 4, !tbaa !7
  br label %32

32:                                               ; preds = %25, %21
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %33 = icmp samesign ult i32 %22, 4
  br i1 %33, label %19, label %21, !llvm.loop !11

34:                                               ; preds = %19
  %35 = zext nneg i32 %5 to i64
  %36 = getelementptr inbounds nuw float, ptr %1, i64 %35
  %37 = load float, ptr addrspacecast (ptr addrspace(3) @_ZZ6reduceE4smem to ptr), align 4, !tbaa !7
  store float %37, ptr %36, align 4, !tbaa !7
  br label %38

38:                                               ; preds = %34, %19
  ret void
}

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

attributes #0 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_89" "target-features"="+ptx87,+sm_89" "uniform-work-group-size"="true" }
attributes #1 = { convergent nocallback nounwind }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!nvvm.annotations = !{!0}
!llvm.module.flags = !{!1, !2, !3}
!llvm.ident = !{!4, !5}
!nvvmir.version = !{!6}

!0 = !{ptr @reduce}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{!"clang version 21.1.8 (https://github.com/conda-forge/clangdev-feedstock 0b2bbeecf482914054e314d49929705c3c8516f8)"}
!5 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!6 = !{i32 2, i32 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"float", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
