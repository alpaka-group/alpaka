# SPDX-FileCopyrightText: Bo Anderson <mail@boanderson.me>
# SPDX-FileCopyrightText: Carlo Cabrera
# SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
# SPDX-FileCopyrightText: Rui Chen <rui@chenrui.dev>
# SPDX-FileCopyrightText: François-Xavier Coudert <fxcoudert@gmail.com>
# SPDX-FileCopyrightText: Raul Ferreira <raulmendesferreira@gmail.com>
# SPDX-FileCopyrightText: Sam Ford
# SPDX-FileCopyrightText: Nanda H. Krishna <nanda.harishankar@gmail.com>
# SPDX-FileCopyrightText: Clément L. <porkepix@gmail.com>
# SPDX-FileCopyrightText: Vai Kong Edward Li
# SPDX-FileCopyrightText: Patrick Linnane <patrick@linnane.io>
# SDPX-FileCopyrightText: Miccal Matthews <miccal.matthews@gmail.com>
# SPDX-FileCopyrightText: Mike McQuaid <mike@mikemcquaid.com>
# SPDX-FileCopyrightText: Sean Molenaar
# SPDX-FileCopyrightText: Rylan Polster <rslpolster@gmail.com>
# SPDX-FileCopyrightText: Michka Popoff <michkapopoff@gmail.com>
# SPDX-FileCopyrightText: Tuta GmbH <https://www.tutao.de>
# SPDX-FileCopyrightText: Caleb Xu <calebcenter@live.com>
# SPDX-FileCopyrightText: Tommy Yang <yt940928@gmail.com>
# SPDX-FileCopyrightText: Ruoyu Zhong <zhongruoyo@outlook.com>
#
# SPDX-FileContributor: Mohammed Anas <triallax@tutanota.com>
# SPDX-FileContributor: Bo Anderson <mail@boanderson.me>
# SPDX-FileContributor: Carlo Cabrera
# SPDX-FileContributor: Rui Chen <rui@chenrui.dev>
# SPDX-FileContributor: Michael Cho <cho-m@tuta.io>
# SPDX-FileContributor: François-Xavier Coudert <fxcoudert@gmail.com>
# SPDX-FileContributor: Raul Ferreira <raulmendesferreira@gmail.com>
# SPDX-FileContributor: Sam Ford
# SPDX-FileContributor: Nanda H. Krishna <nanda.harishankar@gmail.com>
# SPDX-FileContributor: Clément L. <porkepix@gmail.com>
# SPDX-FileContributor: Vai Kong Edward Li
# SPDX-FileContributor: Patrick Linnane <patrick@linnane.io>
# SPDX-FileContributor: Miccal Matthews <miccal.matthews@gmail.com>
# SPDX-FileContributor: Mike McQuaid <mike@mikemcquaid.com>
# SPDX-FileContributor: Sean Molenaar
# SPDX-FileContributor: Rylan Polster <rslpolster@gmail.com>
# SPDX-FileContributor: Michka Popoff <michkapopoff@gmail.com>
# SPDX-FileContributor: Henry Fredrick Schreiner <henry.fredrick.schreiner@cern.ch>
# SPDX-FileContributor: Caleb Xu <calebcenter@live.com>
# SPDX-FileContributor: Tommy Yang <yt940928@gmail.com>
# SPDX-FileContributor: Ruoyu Zhong <zhongruoyo@outlook.com>
# 
# SPDX-License-Identifier: BSD-2-Clause

class Libomp < Formula
  desc "LLVM's OpenMP runtime library"
  homepage "https://openmp.llvm.org/"
  url "https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.7/openmp-15.0.7.src.tar.xz"
  sha256 "3f168d38e7a37b928dcb94b33ce947f75d81eef6fa6a4f9d16b6dc5511c07358"
  license "MIT"

  livecheck do
    url :stable
    regex(/^llvmorg[._-]v?(\d+(?:\.\d+)+)$/i)
  end

  bottle do
    sha256 cellar: :any,                 arm64_ventura:  "8c5c7b912a075e598fb7ae10f2999853343b2662061d92040b1a584cbb3ba7d2"
    sha256 cellar: :any,                 arm64_monterey: "1b1aad07e8677744cdaa264419fade98bd1a852894c77d01985053a96b7d1c7d"
    sha256 cellar: :any,                 arm64_big_sur:  "00e04fbe9783ad7751eaa6d2edda92dfbff85131777255a74e364f3217a7a2df"
    sha256 cellar: :any,                 ventura:        "762c461db6af3cf78983b1eb58aee62699652b96237abf79469c8ac034b2156b"
    sha256 cellar: :any,                 monterey:       "0b944a6bbe8955e7900882b94f1b0b09030d5791191dc5b0c8b3d5d0895f4b12"
    sha256 cellar: :any,                 big_sur:        "f92e5b31f86c22c0fe875b50e050c19a89993b36106a9ad2737230ae2cb68069"
    sha256 cellar: :any_skip_relocation, x86_64_linux:   "d2a16a906c029e8405a11924837417ad1008d41bb1877399f494cb872a179f01"
  end

  # Ref: https://github.com/Homebrew/homebrew-core/issues/112107
  keg_only "it can override GCC headers and result in broken builds"

  depends_on "cmake" => :build
  depends_on "lit" => :build
  uses_from_macos "llvm" => :build

  resource "cmake" do
    url "https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.7/cmake-15.0.7.src.tar.xz"
    sha256 "8986f29b634fdaa9862eedda78513969fe9788301c9f2d938f4c10a3e7a3e7ea"
  end

  def install
    (buildpath/"src").install buildpath.children
    (buildpath/"cmake").install resource("cmake")

    # Disable LIBOMP_INSTALL_ALIASES, otherwise the library is installed as
    # libgomp alias which can conflict with GCC's libgomp.
    args = ["-DLIBOMP_INSTALL_ALIASES=OFF"]
    args << "-DOPENMP_ENABLE_LIBOMPTARGET=OFF" if OS.linux?

    # Build universal binary
    ENV.permit_arch_flags
    ENV.runtime_cpu_detection
    args << "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64"

    system "cmake", "-S", "src", "-B", "build/shared", *std_cmake_args, *args
    system "cmake", "--build", "build/shared"
    system "cmake", "--install", "build/shared"

    system "cmake", "-S", "src", "-B", "build/static",
                    "-DLIBOMP_ENABLE_SHARED=OFF",
                    *std_cmake_args, *args
    system "cmake", "--build", "build/static"
    system "cmake", "--install", "build/static"
  end

  test do
    (testpath/"test.cpp").write <<~EOS
      #include <omp.h>
      #include <array>
      int main (int argc, char** argv) {
        std::array<size_t,2> arr = {0,0};
        #pragma omp parallel num_threads(2)
        {
            size_t tid = omp_get_thread_num();
            arr.at(tid) = tid + 1;
        }
        if(arr.at(0) == 1 && arr.at(1) == 2)
            return 0;
        else
            return 1;
      }
    EOS
    system ENV.cxx, "-Werror", "-Xpreprocessor", "-fopenmp", "test.cpp", "-std=c++11",
                    "-I#{include}", "-L#{lib}", "-lomp", "-o", "test"
    system "./test"
  end
end
