// ==========================================
        // 1. 파티클 네트워크 배경 (Particle Network)
        // ==========================================
        const canvas = document.getElementById('network-canvas');
        const ctx = canvas.getContext('2d');
        let particlesArray;

        const config = {
            particleColor: 'rgba(255, 255, 255, 0.6)',
            lineColor: 'rgba(0, 240, 255,', 
            particleCount: 100, 
            connectionDistance: 150,
            mouseDistance: 200, 
            baseSpeed: 0.5 
        };

        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        let mouse = { x: null, y: null, radius: config.mouseDistance };

        window.addEventListener('mousemove', function(event) {
            mouse.x = event.x;
            mouse.y = event.y;
        });

        class Particle {
            constructor() {
                this.size = (Math.random() * 2) + 1;
                this.x = Math.random() * (innerWidth - this.size * 2) + this.size * 2;
                this.y = Math.random() * (innerHeight - this.size * 2) + this.size * 2;
                this.directionX = (Math.random() * 2) - 1.0;
                this.directionY = (Math.random() * 2) - 1.0;
            }
            update() {
                if (this.x > canvas.width || this.x < 0) this.directionX = -this.directionX;
                if (this.y > canvas.height || this.y < 0) this.directionY = -this.directionY;
                this.x += this.directionX * config.baseSpeed;
                this.y += this.directionY * config.baseSpeed;
            }
            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fillStyle = config.particleColor;
                ctx.fill();
            }
        }

        function init() {
            particlesArray = [];
            let numberOfParticles = (canvas.height * canvas.width) / 15000; 
            for (let i = 0; i < numberOfParticles; i++) {
                particlesArray.push(new Particle());
            }
        }

        function connect() {
            let opacityValue = 1;
            for (let a = 0; a < particlesArray.length; a++) {
                for (let b = a; b < particlesArray.length; b++) {
                    let distance = ((particlesArray[a].x - particlesArray[b].x) ** 2) + 
                                   ((particlesArray[a].y - particlesArray[b].y) ** 2);
                    
                    if (distance < (config.connectionDistance ** 2)) {
                        opacityValue = 1 - (distance / 20000);
                        ctx.strokeStyle = config.lineColor + opacityValue + ')';
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(particlesArray[a].x, particlesArray[a].y);
                        ctx.lineTo(particlesArray[b].x, particlesArray[b].y);
                        ctx.stroke();
                    }
                }
                
                let mouseDistance = ((particlesArray[a].x - mouse.x) ** 2) + 
                                    ((particlesArray[a].y - mouse.y) ** 2);
                
                if (mouseDistance < (config.mouseDistance ** 2)) {
                    opacityValue = 1 - (mouseDistance / 40000);
                    ctx.strokeStyle = 'rgba(0, 240, 255,' + opacityValue + ')';
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(particlesArray[a].x, particlesArray[a].y);
                    ctx.lineTo(mouse.x, mouse.y);
                    ctx.stroke();
                }
            }
        }

        function animate() {
            requestAnimationFrame(animate);
            ctx.clearRect(0, 0, innerWidth, innerHeight);
            for (let i = 0; i < particlesArray.length; i++) {
                particlesArray[i].update();
                particlesArray[i].draw();
            }
            connect();
        }

        window.addEventListener('resize', function() {
            canvas.width = innerWidth;
            canvas.height = innerHeight;
            init();
        });

        window.addEventListener('mouseout', function(){
            mouse.x = undefined;
            mouse.y = undefined;
        });

        // 파티클 시작
        init();
        animate();


        // ==========================================
        // 2. 타이핑 효과 (Typewriter Effect)
        // ==========================================
        const textToType = "Beyond Search, Towards Decision.";
        const typeWriterElement = document.getElementById('typewriter-text');
        const delayedContent = document.getElementById('delayed-content');
        
        let charIndex = 0;
        const typingSpeed = 100; 

        function typeWriter() {
            if (charIndex < textToType.length) {
                typeWriterElement.innerHTML = textToType.substring(0, charIndex + 1) + '<span class="cursor"></span>';
                charIndex++;
                setTimeout(typeWriter, typingSpeed);
            } else {
                typeWriterElement.innerHTML = textToType + '<span class="cursor"></span>';
                setTimeout(() => {
                    delayedContent.classList.add('visible');
                }, 500);
            }
        }

        setTimeout(typeWriter, 500);

        /* ==========================================
           3. 스크롤 애니메이션 (Intersection Observer)
           ========================================== */
        
        // 화면에 요소가 나타나는지 감시하는 감시자 생성
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('active'); // 화면에 들어오면 active 클래스 추가 (CSS에서 등장 효과 실행)
                }
            });
        }, {
            threshold: 0.1 // 요소가 10% 정도 보이면 실행
        });

        // .reveal 클래스를 가진 모든 요소를 감시
        document.querySelectorAll('.reveal').forEach(element => {
            observer.observe(element);
        });