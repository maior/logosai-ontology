/** @type {import('next').NextConfig} */
const nextConfig = {
  devIndicators: false,   // 좌하단 개발자 모드 아이콘 제거
  // 개발 서버와 빌드 서버가 같은 .next 를 쓰면 서로의 산출물을 덮어쓴다
  // (로그의 `__webpack_modules__[moduleId] is not a function` 이 그 증상). dev 는 따로 쓴다.
  distDir: process.env.NEXT_DIST_DIR || '.next',
};
export default nextConfig;
