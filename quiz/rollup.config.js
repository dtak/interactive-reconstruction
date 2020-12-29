import svelte from 'rollup-plugin-svelte';
import resolve from 'rollup-plugin-node-resolve';
import commonjs from 'rollup-plugin-commonjs';
import { terser } from 'rollup-plugin-terser';

const production = !process.env.ROLLUP_WATCH;

export default {
  input: 'src/quiz.js',
  output: {
    sourcemap: true,
    format: 'iife',
    name: 'app',
    file: 'public/quiz.js'
  },
  plugins: [
    svelte({
      skipIntroByDefault: true,
      nestedTransitions: true,
      dev: !production,
      css: css => {
        css.write('public/quiz.css');
      }
    }),
    resolve(),
    commonjs(),
    production && terser()
  ]
};
